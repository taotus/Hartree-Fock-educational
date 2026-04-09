import numpy as np
import math
import scipy.special as sp
from typing import List, Tuple

"""
This module provides primitive Gaussian functions, basis functions (contracted Gaussians),
and utility integrals (Boys function, Gaussian product theorem, overlap integrals) 
for quantum chemistry calculations.

All formulas follow standard references:
- Helgaker, T., Jørgensen, P., & Olsen, J. (2000). Molecular Electronic-Structure Theory.
- Szabo, A., & Ostlund, N. S. (1996). Modern Quantum Chemistry.
"""


# ==================== Primitive Gaussian ====================
class PrimitiveGaussian:
    """
    A primitive Gaussian function (unnormalized) with Cartesian angular momentum.
    A primitive Gaussian is defined as:
        g(x,y,z) = (x - Ax)^lx (y - Ay)^ly (z - Az)^lz * exp(-α |r - A|^2)
    """
    def __init__(self, exponent: float, center: np.ndarray, angular: Tuple[int, int, int] = (0, 0, 0)):
        """
        Parameters
        ----------
        exponent : float
            Exponential factor.
        center : np.ndarray
            Center coordinates [x, y, z].
        angular : Tuple[int, int, int]
            Angular momentum quantum numbers (lx, ly, lz).
        """
        self.exponent = float(exponent)
        self.center = np.asarray(center, dtype=float)
        self.angular = angular  # (lx, ly, lz)

        # Precompute the normalization factor.
        self.norm_factor = self._compute_normalization_factor()

    def _compute_normalization_factor(self) -> float:
        """Compute the normalization factor for the primitive Gaussian."""
        lx, ly, lz = self.angular
        alpha = self.exponent

        # Normalization formula for Gaussian part.
        norm = (2 * alpha / np.pi) ** 0.75
        # For s-type (lx=ly=lz=0), the normalization is just the Gaussian part.
        # For higher angular momentum, we need to include the polynomial part.
        if (lx, ly, lz) == (0, 0, 0):
            return norm
        norm *= (4 * alpha) ** ((lx + ly + lz) / 2)
        norm /= np.sqrt(self._double_factorial(2 * lx - 1) *
                        self._double_factorial(2 * ly - 1) *
                        self._double_factorial(2 * lz - 1))
        return norm

    @staticmethod
    def _double_factorial(n: int) -> float:
        """Compute the double factorial n!!."""
        if n <= 0:
            return 1.0
        result = 1.0
        for i in range(n, 0, -2):
            result *= i
        return result

    def value(self, r: np.ndarray, norm: bool = True) -> float:
        """
        Evaluate the primitive Gaussian at point r.

        Parameters
        ----------
        r : np.ndarray
            Point coordinates.
        norm : bool, optional
            If True, include the normalization factor; otherwise return the unnormalized value.

        Returns
        -------
        float
            Function value.
        """
        x, y, z = r - self.center
        lx, ly, lz = self.angular
        # Gaussian part
        gaussian = np.exp(-self.exponent * (x ** 2 + y ** 2 + z ** 2))
        # Angular part
        polynomial = (x ** lx) * (y ** ly) * (z ** lz)
        if norm:
            return self.norm_factor * polynomial * gaussian
        else:
            return polynomial * gaussian

    def derivative(self, r: np.ndarray, m: int) -> np.ndarray:
        """
        Compute the m-th order derivative of the primitive Gaussian at point r.
        Currently supports m = 1 (first derivative) and m = 2 (second derivative).
        The derivatives are obtained using the recursion:
            d/dx [x^lx e^{-α x^2}] = lx x^{lx-1} e^{-α x^2} - 2α x^{lx+1} e^{-α x^2}

        Parameters
        ----------
        r : np.ndarray
            Point coordinates.
        m : int
            Order of derivative (1 or 2).

        Returns
        -------
        np.ndarray
            Array of derivatives [∂/∂x, ∂/∂y, ∂/∂z] (or second derivatives).
        """
        lx, ly, lz = self.angular

        if m == 1:
            if lx > 0:
                dg_dx = lx * PrimitiveGaussian(
                    self.exponent, self.center, (lx - 1, ly, lz)
                ).value(r, norm=False) - 2.0 * self.exponent * PrimitiveGaussian(
                    self.exponent, self.center, (lx + 1, ly, lz)
                ).value(r, norm=False)
            else:
                dg_dx = - 2.0 * self.exponent * PrimitiveGaussian(
                    self.exponent, self.center, (lx + 1, ly, lz)
                ).value(r, norm=False)
            dg_dx *= self.norm_factor

            if ly > 0:
                dg_dy = ly * PrimitiveGaussian(
                    self.exponent, self.center, (lx, ly - 1, lz)
                ).value(r, norm=False) - 2.0 * self.exponent * PrimitiveGaussian(
                    self.exponent, self.center, (lx, ly + 1, lz)
                ).value(r, norm=False)
            else:
                dg_dy = - 2.0 * self.exponent * PrimitiveGaussian(
                    self.exponent, self.center, (lx, ly + 1, lz)
                ).value(r, norm=False)
            dg_dy *= self.norm_factor

            if lz > 0:
                dg_dz = lz * PrimitiveGaussian(
                    self.exponent, self.center, (lx, ly, lz - 1)
                ).value(r, norm=False) - 2.0 * self.exponent * PrimitiveGaussian(
                    self.exponent, self.center, (lx, ly, lz + 1)
                ).value(r, norm=False)
            else:
                dg_dz = - 2.0 * self.exponent * PrimitiveGaussian(
                    self.exponent, self.center, (lx, ly, lz + 1)
                ).value(r, norm=False)
            dg_dz *= self.norm_factor

        if m == 2:
            dg_dx = 0.
            dg_dx -= 2.0 * self.exponent * (lx + 1) * self.value(r)
            dg_dx += 4.0 * self.exponent ** 2 * PrimitiveGaussian(
                self.exponent, self.center, (lx + 2, ly, lz)
            ).value(r)
            if lx > 0:
                dg_dx += 2 * lx * self.exponent * self.value(r)
                if lx > 1:
                    dg_dx += lx * (lx - 1) * PrimitiveGaussian(
                        self.exponent, self.center, (lx - 2, ly, lz)
                    ).value(r)
            dg_dy = 0.
            dg_dy -= 2.0 * self.exponent * (ly + 1) * self.value(r)
            dg_dy += 4.0 * self.exponent ** 2 * PrimitiveGaussian(
                self.exponent, self.center, (lx, ly + 2, lz)
            ).value(r)
            if ly > 0:
                dg_dy += 2 * ly * self.exponent * self.value(r)
                if ly > 1:
                    dg_dy += ly * (ly - 1) * PrimitiveGaussian(
                        self.exponent, self.center, (lx, ly - 2, lz)
                    ).value(r)
            dg_dz = 0.
            dg_dz -= 2.0 * self.exponent * (lz + 1) * self.value(r)
            dg_dz += 4.0 * self.exponent ** 2 * PrimitiveGaussian(
                self.exponent, self.center, (lx, ly, lz + 2)
            ).value(r)
            if lz > 0:
                dg_dz += 2 * lz * self.exponent * self.value(r)
                if lz > 1:
                    dg_dz += lz * (lz - 1) * PrimitiveGaussian(
                        self.exponent, self.center, (lx, ly, lz - 2)
                    ).value(r)

        else:
            raise NotImplementedError("Derivative class higher than 2 is not supported yet")

        return np.array([dg_dx, dg_dy, dg_dz])


# ==================== Gaussian Integral Utilities ====================
class GaussianIntegral:
    """
    Utility class for Gaussian integrals used in quantum chemistry.

    Provides:
        - Boys function F_n(t) = ∫^1_0 u^{2n} exp(-t u^2) du.
        - One-dimensional Gaussian integrals ∫ x^n exp(-p x^2) dx.
        - Gaussian product theorem: e^{-α|r-A|^2} e^{-β|r-B|^2} = K e^{-p|r-P|^2}.
        - Overlap integrals between two primitive Gaussians.
    """
    @staticmethod
    def boys_function(n: int, t: float) -> float:
        """
        Evaluate the Boys function F_n(t) = ∫_0^1 u^{2n} exp(-t u^2) du.

        For t → 0: F_n(0) = 1/(2n+1).
        For n = 0: F_0(t) = 1/2 * sart(π/t) erf(sqrt(t)).
        For n > 0: recurrence F_n(t) = [ (2n-1) F_{n-1}(t) - e^{-t} ] / (2t).

        Args:
            n: Order of the Boys function.
            t: Argument (usually t = (p q/(p+q)) |P-Q|²).

        Returns:
            float: F_n(t).
        """
        if t < 1e-12:
            return 1.0 / (2 * n + 1)
        if n == 0:
            return 0.5 * np.sqrt(np.pi / t) * sp.erf(np.sqrt(t))
        elif n < 0:
            return 0.
        else:
            return ((2 * n - 1) * GaussianIntegral.boys_function(n - 1, t) -
                    np.exp(-t)) / (2 * t)

    @staticmethod
    def x_n_gaussian_int(n: int, p: float) -> float:
        """
        Evaluate ∫_{-∞}^{∞} x^n exp(-p x^2) dx.

        For odd n: integral = 0.
        For even n: integral = (n! / (2^n (n/2)!)) * sqrt(π / p^{n+1}).

        Args:
            n: Power of x.
            p: Exponent parameter.

        Returns:
            float: Value of the one-dimensional Gaussian integral.
        """
        if n % 2 == 1:
            return 0.
        else:
            return math.factorial(n) / (2**n * math.factorial(int(n/2))) * (np.pi / p ** (n+1))**0.5

    @staticmethod
    def gaussian_product_center(alpha: float, center_a: np.ndarray,
                                beta: float, center_b: np.ndarray) -> Tuple[float, np.ndarray, float]:
        """
        Apply the Gaussian product theorem.

        For two Gaussians centered at A and B:
            exp(-α|r-A|^2) exp(-β|r-B|^2) = K_ab exp(-p|r-P|^2),
        where:
            p = α + β,
            P = (αA + βB) / p,
            K_ab = exp(-αβ|A-B|^2 / p).

        Args:
            alpha: Exponent of first Gaussian.
            center_a: Center of first Gaussian (A).
            beta: Exponent of second Gaussian.
            center_b: Center of second Gaussian (B).

        Returns:
            tuple: (p, center_P, K_ab)
        """
        p = alpha + beta
        center_p = (alpha * center_a + beta * center_b) / p
        r_ab_sq = np.sum((center_a - center_b) ** 2)
        k_factor = np.exp(-alpha * beta * r_ab_sq / p)
        return p, center_p, k_factor

    @staticmethod
    def gaussian_overlap_int(g_a: PrimitiveGaussian, g_b: PrimitiveGaussian) -> float:
        """
        Compute the overlap integral between two primitive Gaussians (unnormalized).

        ∫ g_a(r) g_b(r) dr = K_ab * I_x * I_y * I_z,
        where I_x = ∫ (x-Ax)^lx (x-Bx)^lx' exp(-p (x-Px)^2) dx,
        and similar for y and z. Each I_x is expanded into sums of
        ∫ x^n exp(-p x^2) dx using the binomial theorem.

        Args:
            g_a: First primitive Gaussian.
            g_b: Second primitive Gaussian.

        Returns:
            float: Overlap integral value (without normalization factors).
        """
        # Gaussian product theorem
        alpha, beta = g_a.exponent, g_b.exponent
        center_a, center_b = g_a.center, g_b.center
        p, center_p, k_factor = GaussianIntegral.gaussian_product_center(
            alpha, center_a,
            beta, center_b
        )

        # Angular momenta
        lx1, ly1, lz1 = g_a.angular
        lx2, ly2, lz2 = g_b.angular
        # Shifted coordinates
        x_q, y_q, z_q = center_p - center_a
        x_r, y_r, z_r = center_p - center_b
        int_x, int_y, int_z = 0., 0., 0.

        # x-direction: expand (x - Ax)^lx1 (x - Bx)^lx2
        for i in range(lx1+1):
            for j in range(lx2+1):
                prefactor = sp.comb(lx1, i) * sp.comb(lx2, j)
                prefactor *= x_q ** (lx1 - i) * x_r ** (lx2 - j)
                int_x += prefactor * GaussianIntegral.x_n_gaussian_int(i + j, p)

        # y-direction: expand (y - Ay)^ly1 (y - By)^ly2
        for i in range(ly1+1):
            for j in range(ly2+1):
                prefactor = sp.comb(ly1, i) * sp.comb(ly2, j)
                prefactor *= y_q ** (ly1 - i) * y_r ** (ly2 - j)
                int_y += prefactor * GaussianIntegral.x_n_gaussian_int(i + j, p)

        # z-direction: expand (z - Az)^lz1 (z - Bz)^lz2
        for i in range(lz1+1):
            for j in range(lz2+1):
                prefactor = sp.comb(lz1, i) * sp.comb(lz2, j)
                prefactor *= z_q ** (lz1 - i) * z_r ** (lz2 - j)
                int_z += prefactor * GaussianIntegral.x_n_gaussian_int(i + j, p)

        return k_factor * int_x * int_y * int_z


# ==================== Basis Function ====================
class BasisFunction:
    """
    Contracted Gaussian basis function (contracted Cartesian Gaussian).

    A contracted basis function is a linear combination of primitive Gaussians:
        φ(r) = Σ_i c_i * g_i(r),
    where c_i are contraction coefficients and g_i are primitive Gaussians sharing the same
    center and angular momentum.

    Attributes:
        center (np.ndarray): Center of the basis function.
        coefficients (np.ndarray): Contraction coefficients (c_i).
        exponents (np.ndarray): Exponents (α_i) of primitives.
        angular (Tuple[int,int,int]): Angular momentum (lx,ly,lz) shared by all primitives.
        n_prim (int): Number of primitive Gaussians.
        primitives (List[PrimitiveGaussian]): List of primitive Gaussian objects.
    """

    def __init__(self, center: np.ndarray, coefficients: List[float],
                 exponents: List[float], angular: Tuple[int, int, int] = (0, 0, 0)):
        """
        Initialize a contracted basis function.

        Args:
            center: Center coordinates.
            coefficients: List of contraction coefficients.
            exponents: List of Gaussian exponents.
            angular: Angular momentum (lx, ly, lz).
        """
        self.center = np.asarray(center, dtype=float)
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.exponents = np.asarray(exponents, dtype=float)
        self.angular = angular
        self.n_prim = len(exponents)

        # Create primitive Gaussian objects
        self.primitives = [
            PrimitiveGaussian(exponents[i], center, angular)
            for i in range(self.n_prim)
        ]

    def value(self, r: np.ndarray) -> float:
        """
        Evaluate the contracted basis function at point r.

        φ(r) = Σ_i c_i * g_i(r)

        Args:
            r: Coordinates (x, y, z).

        Returns:
            float: Function value.
        """
        val = 0.0
        for i in range(self.n_prim):
            val += self.coefficients[i] * self.primitives[i].value(r)
        return val

    def squared_derivative(self, r: np.ndarray, m: int) -> float:
        """
        Compute (∂φ/∂x)² + (∂φ/∂y)² + (∂φ/∂z)² for first (m=1) or
        (∂²φ/∂x²)² + (∂²φ/∂y²)² + (∂²φ/∂z²)² for second derivative (m=2).

        Used in kinetic energy density or other gradient-based properties.

        Args:
            r: Coordinates (x, y, z).
            m: Derivative order (1 or 2).

        Returns:
            float: Sum of squares of the derivatives.
        """
        df_dx = 0.0
        df_dy = 0.0
        df_dz = 0.0
        for i in range(self.n_prim):
            df_dx += self.coefficients[i] * self.primitives[i].derivative(r, m)[0]
            df_dy += self.coefficients[i] * self.primitives[i].derivative(r, m)[1]
            df_dz += self.coefficients[i] * self.primitives[i].derivative(r, m)[2]
        return df_dx**2 + df_dy**2 + df_dz**2

    def laplace(self, r: np.ndarray) -> float:
        """
        Compute the Laplacian:
            \nabla^2 \phi(r) = \frac{d^2\phi}{dx^2} + \frac{d^2\phi}{dy^2} + \frac{d^2\phi}{dz^2}

        Used in the kinetic energy operator or in Poisson-type equations.

        Args:
            r: Coordinates (x, y, z).

        Returns:
            float: Laplacian value.
        """
        df_dxx = 0.0
        df_dyy = 0.0
        df_dzz = 0.0
        for i in range(self.n_prim):
            df_dxx += self.coefficients[i] * self.primitives[i].derivative(r, 2)[0]
            df_dyy += self.coefficients[i] * self.primitives[i].derivative(r, 2)[1]
            df_dzz += self.coefficients[i] * self.primitives[i].derivative(r, 2)[2]
        return df_dxx + df_dyy + df_dzz

