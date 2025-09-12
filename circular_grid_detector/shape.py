import math

class Shape:
    def __init__(self, n: int = 0, area: float = 0, cx_sum: float = 0, cy_sum: float = 0, 
                 xx_sum: float = 0, xy_sum: float = 0, yy_sum: float = 0, 
                 cov_xx: float = 0, cov_xy: float = 0, cov_yy: float = 0):
        self.n = n
        self.area = area
        self.cx_sum = cx_sum
        self.cy_sum = cy_sum
        self.xx_sum = xx_sum
        self.xy_sum = xy_sum
        self.yy_sum = yy_sum
        self.cov_xx = cov_xx
        self.cov_xy = cov_xy
        self.cov_yy = cov_yy
        self.kernel_size = 0
        self.block_size = 0
        if area != 0:
            self.x = cx_sum / area
            self.y = cy_sum / area
        else:
            self.x = 0
            self.y = 0
    
    @classmethod
    def from_center_area(cls, cx: float, cy: float, area: float):
        shape = cls()
        shape.x = cx
        shape.y = cy
        shape.area = area
        shape.cx_sum = area * cx
        shape.cy_sum = area * cy
        shape.xx_sum = area / (4 * math.pi)
        shape.xy_sum = 0
        shape.yy_sum = area / (4 * math.pi)
        shape.cov_xx = 0
        shape.cov_xy = 0
        shape.cov_yy = 0
        shape.n = 0
        return shape
    
    def uncertainty(self) -> float:
        det = self.cov_xx * self.cov_yy - self.cov_xy * self.cov_xy
        if det <= 0:
            return float('inf')
        return math.sqrt(det)
    
    def to_string(self) -> str:
        return f"{self.x}\t{self.y}\t{self.cov_xx}\t{self.cov_xy}\t{self.cov_yy}"

class Point2f:
    def __init__(self, x: float = 0, y: float = 0):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point2f(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point2f(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float):
        return Point2f(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float):
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float):
        return Point2f(self.x / scalar, self.y / scalar)

    def dot(self, other) -> float:
        return self.x * other.x + self.y * other.y

    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    def __eq__(self, other):
        return abs(self.x - other.x) < 1e-6 and abs(self.y - other.y) < 1e-6

class Point2i:
    def __init__(self, x: int = 0, y: int = 0):
        self.x = x
        self.y = y

class Size:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def area(self) -> int:
        return self.width * self.height
