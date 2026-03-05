import numpy as np
from typing import List, Dict

from Gaussian import PrimitiveGaussian, GaussianIntegral, BasisFunction
from OSrecursion import ObaraSaikaRecursion


# ==================== One-Electron Integrals ====================
class OneElectronIntegrals:
    """单电子积分计算"""
    @staticmethod
    def overlap_integral(basis_a: BasisFunction, basis_b: BasisFunction) -> float:
        """计算重叠积分 S_ab = ⟨a|b⟩"""
        S = 0.0
        for i in range(basis_a.n_prim):
            for j in range(basis_b.n_prim):
                prim_a = basis_a.primitives[i]
                prim_b = basis_b.primitives[j]
                # 重叠积分值
                integral_value = prim_a.norm_factor * prim_b.norm_factor * GaussianIntegral.gaussian_overlap_int(prim_a, prim_b)
                S += (basis_a.coefficients[i] * basis_b.coefficients[j] * integral_value)

        return S

    @staticmethod
    def kinetic_energy(basis_a: BasisFunction, basis_b: BasisFunction) -> float:
        """计算动能积分 T_ab = ⟨a|-½∇²|b⟩"""
        T = 0.0
        for i in range(basis_a.n_prim):
            for j in range(basis_b.n_prim):
                prim_a = basis_a.primitives[i]
                prim_b = basis_b.primitives[j]

                alpha, beta = prim_a.exponent, prim_b.exponent
                p, center_p, K_ab = GaussianIntegral.gaussian_product_center(
                    alpha, prim_a.center,
                    beta, prim_b.center
                )

                # S轨道动能积分公式
                if prim_a.angular == (0, 0, 0) and prim_b.angular == (0, 0, 0):
                    r_ab_sq = np.sum((prim_a.center - prim_b.center) ** 2)
                    term1 = alpha * beta / p * (3 - 2 * alpha * beta * r_ab_sq / p)
                    term2 = (np.pi / p) ** 1.5 * K_ab

                    integral_value = term1 * term2

                    T += (basis_a.coefficients[i] * basis_b.coefficients[j] *
                          prim_a.norm_factor * prim_b.norm_factor * integral_value)
                else:
                    S_ab = GaussianIntegral.gaussian_overlap_int(prim_a, prim_b)
                    b_x, b_y, b_z = prim_b.angular
                    term1 = 2 * prim_b.exponent * (3 + 2 * (b_x+b_y+b_z)) * S_ab
                    S_a_delta2_b = 0.
                    if b_x >= 2:
                        prim_b_minus_2x = PrimitiveGaussian(
                            prim_b.exponent, prim_b.center, (b_x - 2, b_y, b_z)
                        )
                        S_a_delta2_b += b_x * (b_x - 1) * GaussianIntegral.gaussian_overlap_int(
                            prim_a, prim_b_minus_2x
                        )
                    if b_y >= 2:
                        prim_b_minus_2y = PrimitiveGaussian(
                            prim_b.exponent, prim_b.center, (b_x, b_y - 2, b_z)
                        )
                        S_a_delta2_b += b_y * (b_y - 1) * GaussianIntegral.gaussian_overlap_int(
                            prim_a, prim_b_minus_2y
                        )
                    if b_z >= 2:
                        prim_b_minus_2z = PrimitiveGaussian(
                            prim_b.exponent, prim_b.center, (b_x, b_y, b_z - 2)
                        )
                        S_a_delta2_b += b_z * (b_z - 1) * GaussianIntegral.gaussian_overlap_int(
                            prim_a, prim_b_minus_2z
                        )

                    prim_b_plus_2x = PrimitiveGaussian(
                        prim_b.exponent, prim_b.center, (b_x + 2, b_y, b_z)
                    )
                    prim_b_plus_2y = PrimitiveGaussian(
                        prim_b.exponent, prim_b.center, (b_x, b_y + 2, b_z)
                    )
                    prim_b_plus_2z = PrimitiveGaussian(
                        prim_b.exponent, prim_b.center, (b_x, b_y, b_z + 2)
                    )
                    term2 = 4 * prim_b.exponent**2 * (
                        GaussianIntegral.gaussian_overlap_int(
                            prim_a, prim_b_plus_2x
                        ) +
                        GaussianIntegral.gaussian_overlap_int(
                            prim_a, prim_b_plus_2y
                        ) +
                        GaussianIntegral.gaussian_overlap_int(
                            prim_a, prim_b_plus_2z
                        )
                    )

                    T -= basis_a.coefficients[i] * basis_b.coefficients[j] * (
                        prim_a.norm_factor * prim_b.norm_factor) * 0.5 * (
                        S_a_delta2_b - term1 + term2
                    )

        return T

    @staticmethod
    def nuclear_attraction(basis_a: BasisFunction, basis_b: BasisFunction,
                           nuclei: List[Dict]) -> float:
        """计算核吸引积分 V_ab = ⟨a|∑_A Z_A/|r-R_A||b⟩"""
        V = 0.0
        if basis_a.angular == (0, 0, 0) and basis_b.angular == (0, 0, 0):
            for i in range(basis_a.n_prim):
                for j in range(basis_b.n_prim):
                    prim_a = basis_a.primitives[i]
                    prim_b = basis_b.primitives[j]

                    alpha, beta = prim_a.exponent, prim_b.exponent
                    p, center_p, K_ab = GaussianIntegral.gaussian_product_center(
                        alpha, prim_a.center,
                        beta, prim_b.center
                    )

                    prefactor = (2 * np.pi / p) * K_ab * prim_a.norm_factor * prim_b.norm_factor

                    # 对每个原子核求和
                    for nucleus in nuclei:
                        charge = nucleus["charge"]
                        center_c = np.array(nucleus["position"])

                        t = p * np.sum((center_p - center_c) ** 2)

                        # 使用Boys函数
                        F0 = GaussianIntegral.boys_function(0, t)

                        V -= prefactor * charge * F0 * basis_a.coefficients[i] * basis_b.coefficients[j]

        else:
            for i in range(basis_a.n_prim):
                for j in range(basis_b.n_prim):
                    prim_a = basis_a.primitives[i]
                    prim_b = basis_b.primitives[j]
                    norm_factor = prim_a.norm_factor * prim_b.norm_factor
                    ci = basis_a.coefficients[i]
                    cj = basis_b.coefficients[j]

                    for nucleus in nuclei:
                        V -= nucleus["charge"] * ci * cj * norm_factor * (
                            ObaraSaikaRecursion.nucleus_attraction_integral(prim_a, prim_b, nucleus, 0)
                        )

        return V


# ==================== Two-Electron Integrals ====================
class TwoElectronIntegrals:
    """双电子积分计算"""

    @staticmethod
    def electron_repulsion(basis_a: BasisFunction, basis_b: BasisFunction,
                           basis_c: BasisFunction, basis_d: BasisFunction) -> float:
        """计算电子排斥积分 (ab|cd) = ∫∫ a(r1)b(r1) 1/|r1-r2| c(r2)d(r2) dr1 dr2"""

        integral = 0.0

        # 遍历所有基元组合
        for i in range(basis_a.n_prim):
            for j in range(basis_b.n_prim):
                prim_a = basis_a.primitives[i]
                prim_b = basis_b.primitives[j]

                # AB乘积
                alpha, beta = prim_a.exponent, prim_b.exponent
                p, center_p, K_ab = GaussianIntegral.gaussian_product_center(
                    alpha, prim_a.center,
                    beta, prim_b.center
                )

                for k in range(basis_c.n_prim):
                    for l in range(basis_d.n_prim):
                        prim_c = basis_c.primitives[k]
                        prim_d = basis_d.primitives[l]

                        if prim_a.angular == prim_b.angular == prim_c.angular == prim_d.angular == (0, 0, 0):
                            # CD乘积
                            gamma, delta = prim_c.exponent, prim_d.exponent
                            q, center_q, K_cd = GaussianIntegral.gaussian_product_center(
                                gamma, prim_c.center,
                                delta, prim_d.center
                            )

                            # 计算积分
                            prefactor = (2 * np.pi ** 2.5) / (p * q * np.sqrt(p + q))
                            prefactor *= K_ab * K_cd
                            prefactor *= prim_a.norm_factor * prim_b.norm_factor
                            prefactor *= prim_c.norm_factor * prim_d.norm_factor
                            prefactor *= (basis_a.coefficients[i] * basis_b.coefficients[j] *
                                          basis_c.coefficients[k] * basis_d.coefficients[l])

                            R_pq_sq = np.sum((center_p - center_q) ** 2)

                            # 使用Boys函数
                            t = float(p * q * R_pq_sq / (p + q))
                            F0 = GaussianIntegral.boys_function(0, t)

                            integral += prefactor * F0
                        else:
                            gamma, delta = prim_c.exponent, prim_d.exponent
                            q, center_q, K_cd = GaussianIntegral.gaussian_product_center(
                                gamma, prim_c.center,
                                delta, prim_d.center
                            )
                            norm_factor = (
                                prim_a.norm_factor * prim_b.norm_factor * prim_c.norm_factor * prim_d.norm_factor
                            )
                            ci = basis_a.coefficients[i]
                            cj = basis_b.coefficients[j]
                            ck = basis_c.coefficients[k]
                            cl = basis_d.coefficients[l]
                            prefactor = ci * cj * ck * cl * norm_factor
                            prefactor *= K_ab * K_cd
                            integral += prefactor * ObaraSaikaRecursion.electron_repulsion_integral(
                                prim_a, prim_b, prim_c, prim_d, 0
                            )

        return integral
