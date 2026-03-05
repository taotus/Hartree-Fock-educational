import numpy as np
import math
import scipy.special as sp
from typing import List, Tuple


# ==================== Primitive Gaussian ====================
class PrimitiveGaussian:
    """高斯基元，表示一个未归一化的高斯函数"""

    def __init__(self, exponent: float, center: np.ndarray, angular: Tuple[int, int, int] = (0, 0, 0)):
        """
        Parameters:
        -----------
        exponent : float
            指数因子
        center : np.ndarray
            中心坐标 [x, y, z]
        angular : Tuple[int, int, int]
            角动量量子数 (l, m, n)
        """
        self.exponent = float(exponent)
        self.center = np.asarray(center, dtype=float)
        self.angular = angular  # (lx, ly, lz)

        # 预计算归一化因子
        self.norm_factor = self._compute_normalization_factor()

    def _compute_normalization_factor(self) -> float:
        """计算高斯基元的归一化因子"""
        lx, ly, lz = self.angular
        alpha = self.exponent

        # 归一化因子公式
        norm = (2 * alpha / np.pi) ** 0.75
        if (lx, ly, lz) == (0, 0, 0):
            return norm
        norm *= (4 * alpha) ** ((lx + ly + lz) / 2)
        norm /= np.sqrt(self._double_factorial(2 * lx - 1) *
                        self._double_factorial(2 * ly - 1) *
                        self._double_factorial(2 * lz - 1))
        return norm

    @staticmethod
    def _double_factorial(n: int) -> float:
        """计算双阶乘 (n!!)"""
        if n <= 0:
            return 1.0
        result = 1.0
        for i in range(n, 0, -2):
            result *= i
        return result

    def value(self, r: np.ndarray) -> float:
        """在点r处计算高斯函数值"""
        x, y, z = r - self.center
        lx, ly, lz = self.angular
        # 高斯部分
        gaussian = np.exp(-self.exponent * (x ** 2 + y ** 2 + z ** 2))
        # 角动量部分
        polynomial = (x ** lx) * (y ** ly) * (z ** lz)
        return self.norm_factor * polynomial * gaussian


# ==================== Gaussian Integral Utilities ====================
class GaussianIntegral:
    """高斯积分工具类"""
    @staticmethod
    def boys_function(n: int, t: float) -> float:
        """计算Boys函数 F_n(t)"""
        if t < 1e-12:
            return 1.0 / (2 * n + 1)
        if n == 0:
            return 0.5 * np.sqrt(np.pi / t) * sp.erf(np.sqrt(t))
        elif n < 0:
            return 0.
        else:
            # 递推关系
            return ((2 * n - 1) * GaussianIntegral.boys_function(n - 1, t) -
                    np.exp(-t)) / (2 * t)

    @staticmethod
    def x_n_gaussian_int(n: int, p: float) -> float:
        if n % 2 == 1:
            return 0.
        else:
            return math.factorial(n) / (2**n * math.factorial(int(n/2))) * (np.pi / p ** (n+1))**0.5

    @staticmethod
    def gaussian_product_center(alpha: float, center_a: np.ndarray,
                                beta: float, center_b: np.ndarray) -> Tuple[float, np.ndarray, float]:
        """高斯乘积定理

        Returns:
        --------
        p : float
            乘积高斯函数的指数
        center_p : np.ndarray
            乘积高斯函数的中心
        K_ab : float
            前置因子
        """
        p = alpha + beta
        center_p = (alpha * center_a + beta * center_b) / p
        r_ab_sq = np.sum((center_a - center_b) ** 2)
        k_factor = np.exp(-alpha * beta * r_ab_sq / p)
        return p, center_p, k_factor

    @staticmethod
    def gaussian_overlap_int(g_a: PrimitiveGaussian, g_b: PrimitiveGaussian) -> float:
        """计算两个高斯基元的重叠积分，不包含归一化因子"""
        # 高斯乘积定理
        alpha, beta = g_a.exponent, g_b.exponent
        center_a, center_b = g_a.center, g_b.center
        p, center_p, k_factor = GaussianIntegral.gaussian_product_center(
            alpha, center_a,
            beta, center_b
        )

        # 角动量
        lx1, ly1, lz1 = g_a.angular
        lx2, ly2, lz2 = g_b.angular

        x_q, y_q, z_q = center_p - center_a
        x_r, y_r, z_r = center_p - center_b
        int_x, int_y, int_z = 0., 0., 0.
        # x方向角动量
        for i in range(lx1+1):
            for j in range(lx2+1):
                prefactor = sp.comb(lx1, i) * sp.comb(lx2, j)
                prefactor *= x_q ** (lx1 - i) * x_r ** (lx2 - j)
                int_x += prefactor * GaussianIntegral.x_n_gaussian_int(i + j, p)
        # y方向角动量
        for i in range(ly1+1):
            for j in range(ly2+1):
                prefactor = sp.comb(ly1, i) * sp.comb(ly2, j)
                prefactor *= y_q ** (ly1 - i) * y_r ** (ly2 - j)
                int_y += prefactor * GaussianIntegral.x_n_gaussian_int(i + j, p)
        # z方向角动量
        for i in range(lz1+1):
            for j in range(lz2+1):
                prefactor = sp.comb(lz1, i) * sp.comb(lz2, j)
                prefactor *= z_q ** (lz1 - i) * z_r ** (lz2 - j)
                int_z += prefactor * GaussianIntegral.x_n_gaussian_int(i + j, p)

        return k_factor * int_x * int_y * int_z


# ==================== Basis Function ====================
class BasisFunction:
    """基函数，由多个高斯基元线性组合而成"""

    def __init__(self, center: np.ndarray, coefficients: List[float],
                 exponents: List[float], angular: Tuple[int, int, int] = (0, 0, 0)):
        """
        Parameters:
        -----------
        center : np.ndarray
            基函数中心坐标
        coefficients : List[float]
            每个高斯基元的收缩系数
        exponents : List[float]
            每个高斯基元的指数
        angular : Tuple[int, int, int]
            角动量量子数 (l, m, n)
        """
        self.center = np.asarray(center, dtype=float)
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.exponents = np.asarray(exponents, dtype=float)
        self.angular = angular
        self.n_prim = len(exponents)

        # 创建高斯基元列表
        self.primitives = [
            PrimitiveGaussian(exponents[i], center, angular)
            for i in range(self.n_prim)
        ]

    def value(self, r: np.ndarray) -> float:
        """在点r处计算基函数值"""
        result = 0.0
        for i in range(self.n_prim):
            result += self.coefficients[i] * self.primitives[i].value(r)
        return result
