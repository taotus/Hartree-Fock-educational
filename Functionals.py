import numpy as np


class LDA:
    def __init__(self, weights: np.ndarray):
        self.weights = weights
        self.c_x = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0)

    def compute_ex_vx(self, rho):
        n_points = len(rho)
        e_x = np.zeros(n_points)
        v_x = np.zeros(n_points)

        for g in range(n_points):
            rho_g = rho[g]

            # 避免除零或负数
            if rho_g <= 1e-12:
                e_x[g] = 0.0
                v_x[g] = 0.0
                continue

            # --- 交换部分 (Slater) ---
            # 能量密度 ε_x = C_x * ρ^(1/3)
            rho_13 = rho_g ** (1.0 / 3.0)
            e_x[g] = self.c_x * rho_13

            # 势函数 v_x = (4/3) * C_x * ρ^(1/3) = (4/3) * e_x
            v_x[g] = (4.0 / 3.0) * e_x[g]

        E_x = np.sum(self.weights * (rho * e_x))
        return v_x, E_x

    def compute_ec_vc(self, rho):

        n_points = len(rho)
        e_c = np.zeros(n_points)
        v_c = np.zeros(n_points)

        for g in range(n_points):
            rho_g = rho[g]

            # 避免除零或负数
            if rho_g <= 1e-12:
                e_c[g] = 0.0
                v_c[g] = 0.0
                continue

            # --- 相关部分 (PZ81参数化) ---
            # 计算 r_s
            rs = (3.0 / (4.0 * np.pi * rho_g)) ** (1.0 / 3.0)

            # PZ81相关能密度 ε_c (公式来源: [citation:7])
            if rs >= 1.0:
                sqrt_rs = np.sqrt(rs)
                e_c[g] = -0.1423 / (1.0 + 1.0529 * sqrt_rs + 0.3334 * rs)
            else:
                e_c[g] = -0.0480 + 0.0311 * np.log(rs) - 0.0116 * rs + 0.0020 * rs * np.log(rs)

            # 计算 dε_c/dρ (需要链式法则: dε_c/dρ = dε_c/dr_s * dr_s/dρ)
            # dr_s/dρ = - (1/3) * (3/(4π))^(1/3) * ρ^(-4/3) = - rs / (3ρ)
            if rs >= 1.0:
                # dε_c/dr_s for rs ≥ 1
                denom = (1.0 + 1.0529 * np.sqrt(rs) + 0.3334 * rs)
                de_c_drs = 0.1423 * (0.52645 / np.sqrt(rs) + 0.3334) / (denom * denom)
            else:
                # dε_c/dr_s for rs < 1
                de_c_drs = 0.0311 / rs - 0.0116 + 0.0020 * (np.log(rs) + 1.0)

            # 链式法则: dε_c/dρ = dε_c/dr_s * dr_s/dρ
            drs_drho = -rs / (3.0 * rho_g)
            de_c_drho = de_c_drs * drs_drho

            # 相关势 v_c = ε_c + ρ * (dε_c/dρ)
            v_c[g] = e_c[g] + rho_g * de_c_drho

        # 计算总交换相关能 E_xc = Σ_g w_g * ρ_g * (ε_x + ε_c)
        E_c = np.sum(self.weights * rho * e_c)

        return v_c, E_c


class LYP:
    def __init__(self, weights: np.ndarray):
        self.weights = weights
        self.a = 0.04918
        self.b = 0.132
        self.c = 0.2533
        self.d = 0.349
        self.c_f = 3.0 / 10.0 * (3.0 * np.pi**2) ** (2.0/3.0)

    def _f_1(self, rho: np.ndarray) -> np.ndarray:

        return 1.0 / (1 + self.d * rho ** (-1.0/3.0))

    def _f_1_prime(self, rho: np.ndarray) -> np.ndarray:

        return self._f_1(rho) ** 2 * self.d / 3.0 * rho ** (-4.0/3.0)

    def _e(self, rho: np.ndarray) -> np.ndarray:

        return np.exp(- self.c * rho ** (-1.0 / 3.0))

    def _g_1(self, rho: np.ndarray) -> np.ndarray:

        return self._f_1(rho) * rho ** (-5.0/3.0)

    def _g_1_prime(self, rho: np.ndarray) -> np.ndarray:

        term1 = self._f_1(rho) ** 2 * self._e(rho) * self.d / (3.0 * rho ** (-1.0 / 3.0))
        term2 = - 5.0 / 3.0 * self._f_1(rho) * rho ** (-8.0 / 3.0) * self._e(rho)
        term3 = 1.0 / 3.0 * self._f_1(rho) * rho ** (-3) * self._e(rho)

        return term1 + term2 + term3

    def _g_1_prime2(self, rho: np.ndarray):

        term1 = 2 * self._f_1(rho) * self._f_1_prime(rho) * self.d * self._e(rho) / (3.0 * rho ** 3)
        term2 = - self._f_1(rho) ** 2 * self._e(rho) * self.d / rho ** 4
        term3 = self._f_1(rho) ** 2 * self.c * self.d * self._e(rho) / (9.0 * rho ** 3)
        term4 = - 5 * self._f_1_prime(rho) * self._e(rho) / (3.0 * rho ** (8.0 / 3.0))
        term5 = 40.0 / 9.0 * self._f_1(rho) * self._e(rho) / rho ** (-11.0 / 3.0)
        term6 = - 5.0 * self.c * self._f_1(rho) * self._e(rho) / (9.0 * rho ** 4)
        term7 = self.c * self._f_1_prime(rho) * self._e(rho) / (3.0 * rho ** 3)
        term8 = - self.c * self._f_1(rho) * self._e(rho) / rho ** 4
        term9 = self.c ** 2 * self._f_1(rho) * self._e(rho) / (9.0 * rho ** (-13.0 / 3.0))

        return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9

    def compute_ec_vc(self, rho: np.ndarray, delta_rho: np.ndarray, laplace_rho: np.ndarray):

        # ---------- correlation energy ----------
        t_w = 1.0 / 8.0 * delta_rho / rho
        term1 = self.c_f * rho ** (-2.0/3.0) - 17.0 / 9 * t_w + 1.0 / 18 * laplace_rho

        term2 = rho + 0.132 * rho ** (-2.0 / 3.0) * term1 * np.exp(-self.c * rho ** (-1.0 / 3.0))

        e_c = term2 / (1 + self.d * rho ** (-1.0 / 3.0))

        # ---------- correlation potential ----------
        term3 = - self.a * (self._f_1_prime(rho) * rho + self._f_1(rho))
        term3 -= self.a * self.b * self.c_f * rho ** (5.0 / 3.0) * (self._g_1_prime(rho) + 8.0 / 3.0 * self._g_1(rho))

        term4 = 7.0 / 24.0 * self._g_1_prime2(rho) * rho * delta_rho
        term4 += self._g_1_prime(rho) * (59.0 / 72.0 * delta_rho + 7.0 / 12.0 * rho * laplace_rho)
        term4 += 19.0 / 18.0 * self._g_1(rho) * laplace_rho
        term4 *= self.a * self.b

        v_c = term3 - term4
        E_c = - self.a * np.sum(self.weights * e_c)

        return v_c, E_c
