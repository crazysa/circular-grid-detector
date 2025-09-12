import math
import numpy as np
from numba import jit, prange
from typing import List, Tuple, Optional
from .shape import Shape, Point2i, Point2f
from .utils import intensity_range_directional

class CoreDetection:
    def __init__(self):
        self.numerical_stable = 1e-8
        self.min_size = 100
        self.area_ratio_threshold = 0.02
        self.max_eccentricity = 0.85
        self.distance_threshold = 10
    
    @staticmethod
    @jit(nopython=True)
    def contour_to_shape_properties(contour_x: np.ndarray, contour_y: np.ndarray) -> Tuple[int, float, float, float, float, float, float]:
        n = len(contour_x)
        if n == 0:
            return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        sign = 1
        area, cx_sum, cy_sum, xx_sum, xy_sum, yy_sum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        A, CX, CY, XX, XY, YY = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        for i in range(n):
            x_f = contour_x[i]
            y_f = contour_y[i]
            
            x_c = contour_x[n-1] if i == 0 else contour_x[i-1]
            y_c = contour_y[n-1] if i == 0 else contour_y[i-1]
            
            dxy = x_f * y_c - x_c * y_f
            Mx = x_f + x_c
            My = y_f + y_c
            
            A += dxy
            CX += dxy * Mx
            CY += dxy * My
            XX += dxy * (x_c * Mx + x_f * x_f)
            XY += dxy * (x_c * (My + y_c) + x_f * (My + y_f))
            YY += dxy * (y_c * My + y_f * y_f)
        
        if A < 0:
            sign = -1
        
        area = sign * A / 2
        cx_sum = sign * CX / 6
        cy_sum = sign * CY / 6
        xx_sum = sign * XX / 12
        xy_sum = sign * XY / 24
        yy_sum = sign * YY / 12
        
        return n, area, cx_sum, cy_sum, xx_sum, xy_sum, yy_sum

    def contour_to_shape(self, contour: List[Point2i]) -> Shape:
        contour_x = np.array([p.x for p in contour], dtype=np.float64)
        contour_y = np.array([p.y for p in contour], dtype=np.float64)
        n, area, cx_sum, cy_sum, xx_sum, xy_sum, yy_sum = self.contour_to_shape_properties(contour_x, contour_y)
        return Shape(n, area, cx_sum, cy_sum, xx_sum, xy_sum, yy_sum)
    
    @staticmethod
    @jit(nopython=True)
    def ellipse_test_static(area: float, cx_sum: float, cy_sum: float, xx_sum: float, xy_sum: float, yy_sum: float,
                         min_size: float, area_ratio_threshold: float, max_eccentricity: float) -> bool:
        if area < min_size:
            return False
        
        mx = cx_sum / area
        my = cy_sum / area
        xx = xx_sum / area - mx * mx
        xy = xy_sum / area - mx * my
        yy = yy_sum / area - my * my
        
        det = (xx + yy) * (xx + yy) - 4 * (xx * yy - xy * xy)
        if det > 0:
            det = math.sqrt(det)
        else:
            det = 0
        
        f0 = ((xx + yy) + det) / 2
        f1 = ((xx + yy) - det) / 2
        
        if f0 <= 0 or f1 <= 0:
            return False
        
        m0 = math.sqrt(f0)
        m1 = math.sqrt(f1)
        
        ratio1 = abs(1 - m1 / m0)
        ratio2 = abs(1 - m0 * m1 * 4 * math.pi / area)
        
        if ratio2 > area_ratio_threshold:
            return False
        
        if ratio1 > max_eccentricity:
            return False
        
        return True
    
    def ellipse_test(self, shape: Shape) -> bool:
        return self.ellipse_test_static(shape.area, shape.cx_sum, shape.cy_sum, shape.xx_sum, shape.xy_sum, shape.yy_sum,
                                    self.min_size, self.area_ratio_threshold, self.max_eccentricity)
    
    def calculate_shape_covariance(self, shape: Shape, gray_img: Optional[np.ndarray] = None, grad_x3: Optional[np.ndarray] = None, grad_y3: Optional[np.ndarray] = None) -> bool:
        area = shape.area
        if area <= 0:
            return False
        
        mx = shape.cx_sum / area
        my = shape.cy_sum / area
        xx = shape.xx_sum / area - mx * mx
        xy = shape.xy_sum / area - mx * my
        yy = shape.yy_sum / area - my * my
        
        det = xx * yy - xy * xy
        if det <= 0:
            det = 1e-6
        
        shape.cov_xx = max(xx, 1e-6)
        shape.cov_xy = xy
        shape.cov_yy = max(yy, 1e-6)
        
        return True
    
    
    def cov_to_ellipse(self, cov_xx: float, cov_xy: float, cov_yy: float) -> Tuple[float, float, float]:
        cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        m0 = max(0.0, eigenvals[1])
        m1 = max(0.0, eigenvals[0])
        a = 2 * math.sqrt(m0)
        b = 2 * math.sqrt(m1)
        
        angle = math.atan2(eigenvecs[1, 1], eigenvecs[0, 1]) * 180 / math.pi
        
        return a, b, angle
