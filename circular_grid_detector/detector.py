import math
import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
from .shape import Shape, Point2f, Point2i, Size
from .detection import CoreDetection
from .finder import GridFinder

class GridDetector:
    def __init__(self, n_x: int, n_y: int, is_asymmetric_grid: bool = False):
        self.n_x = n_x
        self.n_y = n_y
        self.is_asymmetric_grid = is_asymmetric_grid
        self.core_detection = CoreDetection()
        self.grid_finder = GridFinder(is_asymmetric_grid)
        self.size_threshold = 100
        self.distance_threshold = 10
        self.workers = min(8, os.cpu_count())
        self.scale_cache = {}
        self.gradient_cache = {}

    def get_parameters(self, img_shape: Tuple[int, int]) -> Tuple[int, int]:
        scale_factor = max(img_shape[0], img_shape[1]) // 1000 + 1
        block_size = 12 * scale_factor + 1
        if block_size % 2 == 0:
            block_size += 1
        kernel_size = max(3, 3 * scale_factor - 1)
        return block_size, kernel_size

    def preprocess_image(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        img_blur = cv2.GaussianBlur(img, (3, 3), 0) if img.shape[0] > 1000 else img
        img_key = id(img)
        if img_key not in self.gradient_cache:
            grad_x = cv2.Scharr(img, cv2.CV_32F, 1, 0) / 32.0
            grad_y = cv2.Scharr(img, cv2.CV_32F, 0, 1) / 32.0
            self.gradient_cache[img_key] = (grad_x, grad_y)
        else:
            grad_x, grad_y = self.gradient_cache[img_key]
        return img_blur, grad_x, grad_y

    def process_threshold(self, args: Tuple) -> List[Tuple[Shape, List[Point2i]]]:
        img_blur, grad_x, grad_y, img_origin, block_size, kernel_size, target_count = args
        img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, block_size, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(img_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        candidates = []
        processed = 0
        contour_areas = [(cv2.contourArea(c), i) for i, c in enumerate(contours)]
        contour_areas.sort(reverse=True)
        for area, idx in contour_areas:
            if processed >= target_count * 2:
                break
            contour = contours[idx]
            if len(contour) < 10:
                continue
            points = [Point2i(int(point[0][0]), int(point[0][1])) for point in contour]
            contour_x = np.array([p.x for p in points], dtype=np.float64)
            contour_y = np.array([p.y for p in points], dtype=np.float64)
            n, area, cx_sum, cy_sum, xx_sum, xy_sum, yy_sum = CoreDetection.contour_to_shape_properties(contour_x, contour_y)
            if CoreDetection.ellipse_test_static(area, cx_sum, cy_sum, xx_sum, xy_sum, yy_sum,
                                             self.core_detection.min_size, 
                                             self.core_detection.area_ratio_threshold, 
                                             self.core_detection.max_eccentricity):
                shape = Shape(n, area, cx_sum, cy_sum, xx_sum, xy_sum, yy_sum)
                if self.core_detection.calculate_shape_covariance(shape):
                    shape.kernel_size = kernel_size
                    shape.block_size = block_size
                    candidates.append((shape, points))
                    processed += 1
        return candidates

    def detect_circles(self, img: np.ndarray) -> Tuple[bool, List[Shape], List[Tuple[Shape, List[Point2i]]]]:
        target_count = self.n_x * self.n_y
        img_blur, grad_x, grad_y = self.preprocess_image(img)
        block_size, kernel_size = self.get_parameters(img.shape)
        args = (img_blur, grad_x, grad_y, img, block_size, kernel_size, target_count * 3)
        circle_candidates = self.process_threshold(args)
        if len(circle_candidates) < target_count * 2:
            block_size_2 = block_size + 4
            if block_size_2 % 2 == 0:
                block_size_2 += 1
            args2 = (img_blur, grad_x, grad_y, img, block_size_2, kernel_size, target_count * 2)
            additional_candidates = self.process_threshold(args2)
            circle_candidates.extend(additional_candidates)
        if len(circle_candidates) < target_count * 2:
            block_size_3 = block_size - 4
            if block_size_3 < 5:
                block_size_3 = 5
            if block_size_3 % 2 == 0:
                block_size_3 += 1
            args3 = (img_blur, grad_x, grad_y, img, block_size_3, kernel_size, target_count * 2)
            additional_candidates_3 = self.process_threshold(args3)
            circle_candidates.extend(additional_candidates_3)
        if circle_candidates:
            areas = np.array([shape.area for shape, _ in circle_candidates])
            sorted_indices = np.argsort(areas)[::-1]
            circle_candidates = [circle_candidates[i] for i in sorted_indices]
        circles = self.remove_duplicates(circle_candidates, target_count)
        source = np.array([[shape.x, shape.y] for shape, _ in circles], dtype=np.float32)
        dest = self.sort_targets(source)
        target = self.match_points_to_shapes(dest, circles)
        result = len(target) == target_count
        return result, target, circles

    def remove_duplicates(self, candidates: List[Tuple[Shape, List[Point2i]]], 
                                   target_count: int) -> List[Tuple[Shape, List[Point2i]]]:
        if not candidates:
            return []
        circles = []
        positions = np.array([[shape.x, shape.y] for shape, _ in candidates])
        uncertainties = np.array([shape.uncertainty() for shape, _ in candidates])
        for i, (shape, contour) in enumerate(candidates):
            if len(circles) >= target_count:
                break
            distances = np.sqrt(np.sum((positions[i] - np.array([[s.x, s.y] for s, _ in circles])) ** 2, axis=1)) if circles else np.array([])
            if len(distances) == 0:
                circles.append((shape, contour))
                continue
            min_distance_idx = np.argmin(distances)
            min_distance = distances[min_distance_idx]
            if min_distance >= self.distance_threshold:
                circles.append((shape, contour))
            elif uncertainties[i] < circles[min_distance_idx][0].uncertainty():
                circles[min_distance_idx] = (shape, contour)
        return circles

    def sort_targets(self, source: np.ndarray) -> List[Point2f]:
        if len(source) == 0:
            return []
        source_points = [Point2f(x, y) for x, y in source]
        if len(source_points) != self.n_x * self.n_y:
            return source_points
        dest = self.grid_sort(source_points)
        if len(dest) == self.n_x * self.n_y:
            return dest
        dest = self.grid_finder.find_grid(source_points, Size(self.n_x, self.n_y))
        return dest

    def grid_sort(self, points: List[Point2f]) -> List[Point2f]:
        if len(points) != self.n_x * self.n_y:
            return []
        points_array = np.array([[p.x, p.y] for p in points])
        y_values = points_array[:, 1]
        y_sorted_indices = np.argsort(y_values)
        y_sorted = y_values[y_sorted_indices]
        y_gaps = []
        for i in range(1, len(y_sorted)):
            gap = y_sorted[i] - y_sorted[i-1]
            y_gaps.append(gap)
        y_gaps = np.array(y_gaps)
        gap_threshold = max(np.percentile(y_gaps, 90), 20)
        large_gaps = y_gaps > gap_threshold
        row_breaks = [0]
        for i, is_large_gap in enumerate(large_gaps):
            if is_large_gap:
                row_breaks.append(i + 1)
        row_breaks.append(len(points))
        if len(row_breaks) - 1 > self.n_y:
            sorted_gap_indices = np.argsort(y_gaps)[::-1]
            best_breaks = None
            best_score = float('inf')
            for num_gaps in range(min(len(sorted_gap_indices), 15), 3, -1):
                test_breaks = [0]
                for i in range(self.n_y - 1):
                    if i < num_gaps:
                        test_breaks.append(sorted_gap_indices[i] + 1)
                test_breaks.append(len(points))
                test_breaks.sort()
                if len(test_breaks) - 1 == self.n_y:
                    score = 0
                    valid = True
                    for j in range(len(test_breaks) - 1):
                        row_size = test_breaks[j + 1] - test_breaks[j]
                        score += abs(row_size - self.n_x)
                        if row_size < self.n_x // 2 or row_size > self.n_x * 2:
                            valid = False
                            break
                    if valid and score < best_score:
                        best_score = score
                        best_breaks = test_breaks
            if best_breaks is not None:
                row_breaks = best_breaks
            else:
                return []
        if len(row_breaks) - 1 != self.n_y:
            return []
        result = []
        for i in range(len(row_breaks) - 1):
            start_idx = row_breaks[i]
            end_idx = row_breaks[i + 1]
            row_size = end_idx - start_idx
            if row_size < self.n_x // 2 or row_size > self.n_x * 2:
                return []
            row_indices = y_sorted_indices[start_idx:end_idx]
            row_points = [points[idx] for idx in row_indices]
            row_points.sort(key=lambda p: p.x)
            if row_size == self.n_x:
                result.extend(row_points)
            elif row_size < self.n_x:
                result.extend(row_points)
                for _ in range(self.n_x - row_size):
                    result.append(Point2f(-1, -1))
            else:
                result.extend(row_points[:self.n_x])
        if len([p for p in result if p.x >= 0]) < self.n_x * self.n_y * 0.8:
            return []
        return [p for p in result if p.x >= 0]

    def match_points_to_shapes(self, dest: List[Point2f], 
                                        circles: List[Tuple[Shape, List[Point2i]]]) -> List[Shape]:
        if not dest or not circles:
            return []
        target = []
        dest_positions = np.array([[p.x, p.y] for p in dest])
        circle_positions = np.array([[shape.x, shape.y] for shape, _ in circles])
        for dest_pos in dest_positions:
            distances = np.sqrt(np.sum((circle_positions - dest_pos) ** 2, axis=1))
            min_idx = np.argmin(distances)
            if distances[min_idx] < self.distance_threshold:
                target.append(circles[min_idx][0])
        return target

    def create_visualization_folder(self, img_path: str) -> str:
        img_dir = os.path.dirname(img_path)
        viz_dir = os.path.join(img_dir, "visualization")
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        return viz_dir

    def create_visualization(self, img: np.ndarray, success: bool, centers: List[Point2f], 
                           all_candidates: List[Tuple[Shape, List[Point2i]]], 
                           img_path: str) -> np.ndarray:
        if len(img.shape) == 2:
            vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = img.copy()
        if success:
            for i, center in enumerate(centers):
                cv2.circle(vis_img, (int(center.x), int(center.y)), 5, (0, 255, 0), 2)
                cv2.putText(vis_img, str(i), (int(center.x) + 10, int(center.y) + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            status_text = f"SUCCESS: {len(centers)}/{self.n_x * self.n_y}"
            color = (0, 255, 0)
        else:
            for i, (shape, contour) in enumerate(all_candidates):
                contour_points = [(p.x, p.y) for p in contour]
                cv2.drawContours(vis_img, [np.array(contour_points, dtype=np.int32)], -1, (255, 0, 0), cv2.LINE_4)
                cv2.circle(vis_img, (int(shape.x), int(shape.y)), 3, (0, 255, 255), -1)
            status_text = f"DETECTION FAIL: {len(centers)}/{self.n_x * self.n_y}"
            color = (0, 0, 255)
        cv2.putText(vis_img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(vis_img, f"Candidates: {len(all_candidates)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return vis_img

    def detect(self, img_path: str, debug: bool = False) -> Tuple[bool, List[Point2f]]:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        original_shape = img.shape
        if max(img.shape) > 2000:
            scale = 2000 / max(img.shape)
            new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            scale_back = True
        else:
            scale = 1.0
            scale_back = False
        if img.shape[0] > 1000 or img.shape[1] > 1000:
            img = cv2.bilateralFilter(img, 5, 80, 80)
        success, shapes, all_candidates = self.detect_circles(img)
        centers = []
        if success:
            for shape in shapes:
                if scale_back:
                    x = shape.x / scale
                    y = shape.y / scale
                    centers.append(Point2f(x, y))
                else:
                    centers.append(Point2f(shape.x, shape.y))
        if debug:
            if scale_back:
                vis_img_base = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                scaled_candidates = []
                for shape, contour in all_candidates:
                    scaled_shape = Shape(shape.n, shape.area, shape.cx_sum/scale, shape.cy_sum/scale, 
                                       shape.xx_sum, shape.xy_sum, shape.yy_sum)
                    scaled_shape.x = shape.x / scale
                    scaled_shape.y = shape.y / scale
                    scaled_contour = [Point2i(int(p.x / scale), int(p.y / scale)) for p in contour]
                    scaled_candidates.append((scaled_shape, scaled_contour))
                vis_candidates = scaled_candidates
            else:
                vis_img_base = img
                vis_candidates = all_candidates
            vis_img = self.create_visualization(vis_img_base, success, centers, vis_candidates, img_path)
            viz_dir = self.create_visualization_folder(img_path)
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            vis_path = os.path.join(viz_dir, f"{img_name}_detection.jpg")
            cv2.imwrite(vis_path, vis_img)
        if len(self.gradient_cache) > 10:
            self.gradient_cache.clear()
        return success, centers

def detect_grid(img_path: str, grid_width: int, grid_height: int,
               is_asymmetric: bool = False, debug: bool = False) -> Tuple[bool, List[Tuple[float, float]]]:
    detector = GridDetector(grid_width, grid_height, is_asymmetric)
    success, centers = detector.detect(img_path, debug)
    center_tuples = [(p.x, p.y) for p in centers]
    return success, center_tuples
