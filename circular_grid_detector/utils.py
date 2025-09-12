import math
import numpy as np
import cv2
from numba import jit, prange
from typing import List, Tuple, Optional
from .shape import Point2i, Point2f

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return img

def preprocessing(img: np.ndarray, detection_mode: str = "gray") -> np.ndarray:
    if len(img.shape) == 3:
        if detection_mode == "saturation":
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            return hsv[:, :, 1]
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        return cv2.bilateralFilter(img, -1, 10, 10)
    return img

def scharr_gradient(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    grad_x = cv2.Scharr(img, cv2.CV_32F, 1, 0) / 32.0
    grad_y = cv2.Scharr(img, cv2.CV_32F, 0, 1) / 32.0
    return grad_x, grad_y

def gaussian_blur(img: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def adaptive_threshold(img: np.ndarray, max_val: int, block_size: int, C: int) -> np.ndarray:
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(img, max_val, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C)

def morphology_close(img: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def find_contours(img: np.ndarray) -> List[List[Point2i]]:
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    result = []
    
    for contour in contours:
        if len(contour) >= 10:
            points = []
            for point in contour:
                x, y = point[0]
                points.append(Point2i(int(x), int(y)))
            result.append(points)
    
    return result


def convex_hull(points: List[Point2f]) -> List[Point2f]:
    if len(points) < 3:
        return points
    
    pts = np.array([[p.x, p.y] for p in points], dtype=np.float32)
    hull = cv2.convexHull(pts)
    hull_points = []
    for point in hull:
        hull_points.append(Point2f(point[0][0], point[0][1]))
    
    return hull_points


def find_homography(src_points: List[Point2f], dst_points: List[Point2f]) -> np.ndarray:
    if len(src_points) < 4:
        return np.eye(3, dtype=np.float32)
    
    src = np.array([[p.x, p.y] for p in src_points], dtype=np.float32)
    dst = np.array([[p.x, p.y] for p in dst_points], dtype=np.float32)
    
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        return np.eye(3, dtype=np.float32)
    return H


def transform_points(points: List[Point2f], homography: np.ndarray) -> List[Point2f]:
    if len(points) == 0:
        return []
    
    pts = np.array([[p.x, p.y] for p in points], dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, homography)
    
    result = []
    for point in transformed:
        x, y = point[0]
        result.append(Point2f(x, y))
    
    return result


@jit(nopython=True, parallel=True)
def intensity_range_directional_fast(gray_img: np.ndarray, x: int, y: int, 
                                   step_x: float, step_y: float, step_length: int) -> int:
    height, width = gray_img.shape
    min_val = 255
    max_val = 0
    
    for i in prange(-step_length, step_length + 1):
        xi = int(x + round(step_x * i))
        yi = int(y + round(step_y * i))
        xi = max(0, min(xi, width - 1))
        yi = max(0, min(yi, height - 1))
        
        value = gray_img[yi, xi]
        if value > max_val:
            max_val = value
        if value < min_val:
            min_val = value
    
    return max_val - min_val


@jit(nopython=True, parallel=True)
def intensity_range_window_fast(gray_img: np.ndarray, x: int, y: int, window_size: int) -> int:
    height, width = gray_img.shape
    min_val = 255
    max_val = 0
    
    for i in prange(-window_size, window_size + 1):
        for j in prange(-window_size, window_size + 1):
            xi = max(0, min(x + i, width - 1))
            yi = max(0, min(y + j, height - 1))
            
            value = gray_img[yi, xi]
            if value > max_val:
                max_val = value
            if value < min_val:
                min_val = value
    
    return max_val - min_val


def intensity_range_directional(gray_img: np.ndarray, x: int, y: int, 
                              step_x: float, step_y: float, step_length: int) -> int:
    return intensity_range_directional_fast(gray_img, x, y, step_x, step_y, step_length)


def intensity_range_window(gray_img: np.ndarray, x: int, y: int, window_size: int) -> int:
    return intensity_range_window_fast(gray_img, x, y, window_size)
