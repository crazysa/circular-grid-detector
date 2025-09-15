import math
import numpy as np
from numba import jit, prange
from typing import List, Tuple
from .shape import Point2f, Size
from .utils import convex_hull, find_homography, transform_points

class PointIndex:
    def __init__(self, points: List[Point2f]):
        self.points = np.array([[p.x, p.y] for p in points], dtype=np.float32)
    
    def nearest_search(self, query_point: Point2f, k: int = 1) -> Tuple[List[int], List[float]]:
        query = np.array([query_point.x, query_point.y], dtype=np.float32)
        distances = np.sum((self.points - query) ** 2, axis=1)
        indices = np.argsort(distances)[:k]
        dists = distances[indices]
        return indices.tolist(), dists.tolist()

class GridFinder:
    def __init__(self, is_asymmetric_grid: bool):
        self.is_asymmetric_grid = is_asymmetric_grid
        self.square_size = 1.0
        self.max_rectified_distance = self.square_size / 2.0
        self.pattern_size = None
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def compute_distances(points_x: np.ndarray, points_y: np.ndarray) -> np.ndarray:
        n = len(points_x)
        dists = np.zeros((n, n), dtype=np.float32)
        
        for i in prange(n):
            for j in prange(i + 1, n):
                dist = math.sqrt((points_x[i] - points_x[j]) ** 2 + (points_y[i] - points_y[j]) ** 2)
                dists[i, j] = dist
                dists[j, i] = dist
        
        return dists
    
    def cluster_points(self, points: List[Point2f], pattern_sz: Size) -> List[Point2f]:
        pattern_points = []
        n = len(points)
        pn = pattern_sz.area()
        
        if pn >= n:
            if pn == n:
                pattern_points = points[:]
            return pattern_points
        
        points_x = np.array([p.x for p in points], dtype=np.float32)
        points_y = np.array([p.y for p in points], dtype=np.float32)
        
        dists = self.compute_distances(points_x, points_y)
        dists_mask = np.ones_like(dists, dtype=np.uint8)
        np.fill_diagonal(dists_mask, 0)
        
        clusters = [set([i]) for i in range(n)]
        
        pattern_cluster_idx = 0
        while len(clusters[pattern_cluster_idx]) < pn:
            masked_dists = np.where(dists_mask == 1, dists, np.inf)
            min_indices = np.unravel_index(np.argmin(masked_dists), masked_dists.shape)
            min_idx = min(min_indices[0], min_indices[1])
            max_idx = max(min_indices[0], min_indices[1])
            
            dists_mask[max_idx, :] = 0
            dists_mask[:, max_idx] = 0
            
            for k in range(n):
                dists[min_idx, k] = min(dists[min_idx, k], dists[max_idx, k])
                dists[k, min_idx] = dists[min_idx, k]
            
            clusters[min_idx].update(clusters[max_idx])
            clusters[max_idx] = set()
            pattern_cluster_idx = min_idx
        
        if len(clusters[pattern_cluster_idx]) != pn:
            return []
        
        pattern_points = []
        for idx in clusters[pattern_cluster_idx]:
            pattern_points.append(points[idx])
        
        return pattern_points
    
    def find_corners(self, hull2f: List[Point2f]) -> List[Point2f]:
        angles = []
        n = len(hull2f)
        
        for i in range(n):
            vec1 = hull2f[(i + 1) % n] - hull2f[i]
            vec2 = hull2f[(i - 1 + n) % n] - hull2f[i]
            
            norm1 = vec1.norm()
            norm2 = vec2.norm()
            
            if norm1 > 1e-6 and norm2 > 1e-6:
                angle = vec1.dot(vec2) / (norm1 * norm2)
            else:
                angle = 0
            
            angles.append(angle)
        
        sorted_indices = sorted(range(len(angles)), key=lambda i: angles[i], reverse=True)
        corners_count = 6 if self.is_asymmetric_grid else 4
        corner_indices = sorted(sorted_indices[:corners_count])
        
        corners = []
        for i in corner_indices:
            corners.append(hull2f[i])
        
        return corners
    
    def find_outside_corners(self, corners: List[Point2f]) -> List[Point2f]:
        if len(corners) == 0:
            return []
        
        outside_corners = []
        n = len(corners)
        
        tangent_vectors = []
        for k in range(n):
            diff = corners[(k + 1) % n] - corners[k]
            norm_val = diff.norm()
            if norm_val > 1e-6:
                tangent_vectors.append(diff / norm_val)
            else:
                tangent_vectors.append(Point2f(0, 0))
        
        cos_angles = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                val = abs(tangent_vectors[i].dot(tangent_vectors[j]))
                cos_angles[i, j] = val
                cos_angles[j, i] = val
        
        max_loc = np.unravel_index(np.argmax(cos_angles), cos_angles.shape)
        diff_between_false_lines = 3
        
        if abs(max_loc[0] - max_loc[1]) == diff_between_false_lines:
            cos_angles[max_loc[0], :] = 0.0
            cos_angles[:, max_loc[0]] = 0.0
            cos_angles[max_loc[1], :] = 0.0
            cos_angles[:, max_loc[1]] = 0.0
            max_loc = np.unravel_index(np.argmax(cos_angles), cos_angles.shape)
        
        max_idx = max(max_loc[0], max_loc[1])
        min_idx = min(max_loc[0], max_loc[1])
        big_diff = 4
        
        if max_idx - min_idx == big_diff:
            min_idx += n
            max_idx, min_idx = min_idx, max_idx
        
        if max_idx - min_idx != n - big_diff:
            return []
        
        outsiders_segment_idx = (min_idx + max_idx) // 2
        
        outside_corners.append(corners[outsiders_segment_idx % n])
        outside_corners.append(corners[(outsiders_segment_idx + 1) % n])
        
        return outside_corners
    
    def point_line_distance(self, p: Point2f, line: Tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = line
        pa = np.array([x1, y1, 1])
        pb = np.array([x2, y2, 1])
        line_vec = np.cross(pa, pb)
        
        if abs(line_vec[0]) < 1e-8 and abs(line_vec[1]) < 1e-8:
            return 0.0
        
        return abs(p.x * line_vec[0] + p.y * line_vec[1] + line_vec[2]) / math.sqrt(line_vec[0] * line_vec[0] + line_vec[1] * line_vec[1])
    
    def get_sorted_corners(self, hull2f: List[Point2f], pattern_points: List[Point2f], 
                          corners: List[Point2f], outside_corners: List[Point2f]) -> List[Point2f]:
        first_corner = Point2f()
        
        if self.is_asymmetric_grid:
            center = Point2f(0, 0)
            for corner in corners:
                center = center + corner
            center = center / len(corners)
            
            center_to_corners = []
            for outside_corner in outside_corners:
                center_to_corners.append(outside_corner - center)
            
            if len(center_to_corners) >= 2:
                cross_product = (center_to_corners[0].x * center_to_corners[1].y - 
                               center_to_corners[0].y * center_to_corners[1].x)
                is_clockwise = cross_product > 0
                first_corner = outside_corners[1] if is_clockwise else outside_corners[0]
            else:
                first_corner = corners[0]
        else:
            first_corner = corners[0]
        
        first_corner_idx = -1
        for i, hull_point in enumerate(hull2f):
            if abs(hull_point.x - first_corner.x) < 1e-6 and abs(hull_point.y - first_corner.y) < 1e-6:
                first_corner_idx = i
                break
        
        sorted_corners = []
        n_hull = len(hull2f)
        
        for i in range(n_hull):
            idx = (first_corner_idx + i) % n_hull
            hull_point = hull2f[idx]
            
            for corner in corners:
                if abs(corner.x - hull_point.x) < 1e-6 and abs(corner.y - hull_point.y) < 1e-6:
                    sorted_corners.append(corner)
                    break
        
        if not self.is_asymmetric_grid and len(sorted_corners) >= 3:
            dist01 = (sorted_corners[0] - sorted_corners[1]).norm()
            dist12 = (sorted_corners[1] - sorted_corners[2]).norm()
            thresh = min(dist01, dist12) / min(self.pattern_size.width, self.pattern_size.height) / 2
            
            circle_count01 = 0
            circle_count12 = 0
            line01 = (sorted_corners[0].x, sorted_corners[0].y, sorted_corners[1].x, sorted_corners[1].y)
            line12 = (sorted_corners[1].x, sorted_corners[1].y, sorted_corners[2].x, sorted_corners[2].y)
            
            for pattern_point in pattern_points:
                if self.point_line_distance(pattern_point, line01) < thresh:
                    circle_count01 += 1
                if self.point_line_distance(pattern_point, line12) < thresh:
                    circle_count12 += 1
            
            if ((circle_count01 > circle_count12 and self.pattern_size.height > self.pattern_size.width) or
                (circle_count01 < circle_count12 and self.pattern_size.height < self.pattern_size.width)):
                sorted_corners = sorted_corners[1:] + [sorted_corners[0]]
        
        return sorted_corners
    
    def rectify_pattern_points(self, pattern_points: List[Point2f], sorted_corners: List[Point2f]) -> List[Point2f]:
        true_indices = [
            (0, 0),
            (self.pattern_size.width - 1, 0),
        ]
        
        if self.is_asymmetric_grid:
            true_indices.extend([
                (self.pattern_size.width - 1, 1),
                (self.pattern_size.width - 1, self.pattern_size.height - 2),
            ])
        
        true_indices.extend([
            (self.pattern_size.width - 1, self.pattern_size.height - 1),
            (0, self.pattern_size.height - 1),
        ])
        
        ideal_points = []
        for j, i in true_indices:
            if self.is_asymmetric_grid:
                ideal_x = (2 * j + i % 2) * self.square_size
                ideal_y = i * self.square_size
            else:
                ideal_x = j * self.square_size
                ideal_y = i * self.square_size
            ideal_points.append(Point2f(ideal_x, ideal_y))
        
        homography = find_homography(sorted_corners, ideal_points)
        rectified_pattern_points = transform_points(pattern_points, homography)
        
        return rectified_pattern_points
    
    def parse_pattern_points(self, pattern_points: List[Point2f], rectified_pattern_points: List[Point2f]) -> List[Point2f]:
        if len(rectified_pattern_points) == 0:
            return []
        
        point_index = PointIndex(rectified_pattern_points)
        centers = []
        used_indices = set()
        
        for i in range(self.pattern_size.height):
            for j in range(self.pattern_size.width):
                if self.is_asymmetric_grid:
                    ideal_pt = Point2f((2 * j + i % 2) * self.square_size, i * self.square_size)
                else:
                    ideal_pt = Point2f(j * self.square_size, i * self.square_size)
                
                k = min(5, len(rectified_pattern_points))
                indices, dists = point_index.nearest_search(ideal_pt, k)
                
                found_valid = False
                for idx_pos in range(len(indices)):
                    candidate_idx = indices[idx_pos]
                    if candidate_idx not in used_indices:
                        if dists[idx_pos] <= self.max_rectified_distance:
                            centers.append(pattern_points[candidate_idx])
                            used_indices.add(candidate_idx)
                            found_valid = True
                            break
                        else:
                            return []
                
                if not found_valid:
                    return []
        
        return centers
    
    def find_grid(self, points: List[Point2f], pattern_size: Size) -> List[Point2f]:
        self.pattern_size = pattern_size
        centers = []
        
        if len(points) == 0:
            return centers
        
        pattern_points = self.cluster_points(points, pattern_size)
        if len(pattern_points) == 0:
            return centers
        
        hull2f = convex_hull(pattern_points)
        corners_count = 6 if self.is_asymmetric_grid else 4
        if len(hull2f) < corners_count:
            return centers
        
        corners = self.find_corners(hull2f)
        if len(corners) != corners_count:
            return centers
        
        outside_corners = []
        if self.is_asymmetric_grid:
            outside_corners = self.find_outside_corners(corners)
            if len(outside_corners) != 2:
                return centers
        
        sorted_corners = self.get_sorted_corners(hull2f, pattern_points, corners, outside_corners)
        if len(sorted_corners) != corners_count:
            return centers
        
        rectified_pattern_points = self.rectify_pattern_points(pattern_points, sorted_corners)
        if len(pattern_points) != len(rectified_pattern_points):
            return centers
        
        centers = self.parse_pattern_points(pattern_points, rectified_pattern_points)
        return centers
