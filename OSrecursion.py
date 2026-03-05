import numpy as np

from Gaussian import PrimitiveGaussian, GaussianIntegral


# ==================== Obara–Saika Integrals ====================
class ObaraSaikaRecursion:
    @staticmethod
    def nucleus_attraction_integral(g_a: PrimitiveGaussian, g_b: PrimitiveGaussian, nucleus: dict, m: int):

        if g_a.angular == (0, 0, 0) and g_b.angular == (0, 0, 0):
            p, center_p, k_factor = GaussianIntegral.gaussian_product_center(
                g_a.exponent, g_a.center,
                g_b.exponent, g_b.center
            )

            center_c = nucleus["position"]
            t = float(p * np.sum((center_p - center_c) ** 2))
            integral_value = 2 * np.pi / p * GaussianIntegral.boys_function(m, t)

            return k_factor * integral_value

        elif any(a < 0 for a in g_a.angular) or any(b < 0 for b in g_b.angular):
            return 0

        elif any(a > 0 for a in g_a.angular):
            p, center_p, k_factor = GaussianIntegral.gaussian_product_center(
                g_a.exponent, g_a.center,
                g_b.exponent, g_b.center
            )

            for cord in range(3):
                if g_a.angular[cord] == 0:
                    continue
                deltal = tuple(1 if idx == cord else 0 for idx in range(3))
                angular_a1 = tuple(a - l for a, l in zip(g_a.angular, deltal))
                angular_a2 = tuple(a - 2 * l for a, l in zip(g_a.angular, deltal))
                angular_b1 = tuple(b - l for b, l in zip(g_b.angular, deltal))
                center_c = nucleus["position"]
                g_a1 = PrimitiveGaussian(
                    g_a.exponent, g_a.center, angular_a1
                )
                g_a2 = PrimitiveGaussian(
                    g_a.exponent, g_a.center, angular_a2
                )
                g_b1 = PrimitiveGaussian(
                    g_b.exponent, g_b.center, angular_b1
                )
                integral = (center_p[cord] - g_a.center[cord]) * ObaraSaikaRecursion.nucleus_attraction_integral(
                    g_a1, g_b, nucleus, m
                )
                integral += 1 / (2 * p) * (
                    (g_a.angular[cord] - 1) * ObaraSaikaRecursion.nucleus_attraction_integral(
                        g_a2, g_b, nucleus, m
                    ) + g_b.angular[cord] * ObaraSaikaRecursion.nucleus_attraction_integral(
                        g_a1, g_b1, nucleus, m
                    )
                )
                integral -= 1 / (2 * p) * (
                    (g_a.angular[cord] - 1) * ObaraSaikaRecursion.nucleus_attraction_integral(
                        g_a2, g_b, nucleus,m + 1
                    ) + g_b.angular[cord] * ObaraSaikaRecursion.nucleus_attraction_integral(
                        g_a1, g_b1, nucleus, m + 1
                    )
                )
                integral -= (center_p[cord] - center_c[cord]) * ObaraSaikaRecursion.nucleus_attraction_integral(
                    g_a1, g_b, nucleus, m + 1
                )
                return integral
        elif any(b > 0 for b in g_b.angular):
            return ObaraSaikaRecursion.nucleus_attraction_integral(g_b, g_a, nucleus, m)

    @staticmethod
    def electron_repulsion_integral(g_a: PrimitiveGaussian, g_b: PrimitiveGaussian,
                                    g_c: PrimitiveGaussian, g_d: PrimitiveGaussian, m: int):
        #print(f"a: {g_a.angular}, b: {g_b.angular}, c: {g_c.angular}, d: {g_d.angular}")
        if any(l < 0 for l in (g_a.angular + g_b.angular + g_c.angular + g_d.angular)):
            return 0

        elif g_b.angular == g_d.angular == (0, 0, 0):  # 垂直递归

            if g_a.angular == g_c.angular == (0, 0, 0):  # 递归原点
                p, center_p, k_ab = GaussianIntegral.gaussian_product_center(
                    g_a.exponent, g_a.center,
                    g_b.exponent, g_b.center
                )
                q, center_q, k_cd = GaussianIntegral.gaussian_product_center(
                    g_c.exponent, g_c.center,
                    g_d.exponent, g_d.center
                )
                prefactor = (2 * np.pi ** 2.5) / (p * q * np.sqrt(p + q))

                t = float(p * q * np.sum((center_p - center_q) ** 2) / (p + q))
                return prefactor * GaussianIntegral.boys_function(m, t)
            elif any(a > 0 for a in g_a.angular):

                p, center_p, k_ab = GaussianIntegral.gaussian_product_center(
                    g_a.exponent, g_a.center,
                    g_b.exponent, g_b.center
                )
                q, center_q, k_cd = GaussianIntegral.gaussian_product_center(
                    g_c.exponent, g_c.center,
                    g_d.exponent, g_d.center
                )

                for i in range(3):
                    if g_a.angular[i] == 0:
                        continue
                    deltal = tuple(1 if idx == i else 0 for idx in range(3))

                    angular_a1 = tuple(a - l for a, l in zip(g_a.angular, deltal))
                    angular_a2 = tuple(a - 2 * l for a, l in zip(g_a.angular, deltal))
                    angular_c1 = tuple(c - l for c, l in zip(g_c.angular, deltal))
                    ga_1 = PrimitiveGaussian(
                        g_a.exponent, g_a.center, angular_a1
                    )
                    gc_1 = PrimitiveGaussian(
                        g_c.exponent, g_c.center, angular_c1
                    )
                    ga_2 = PrimitiveGaussian(
                        g_a.exponent, g_a.center, angular_a2
                    )
                    integral = (center_p[i] - ga_1.center[i]) * (
                        ObaraSaikaRecursion.electron_repulsion_integral(ga_1, g_b, g_c, g_d, m)
                    )
                    integral -= q / (p + q) * (center_p[i] - center_q[i]) * (
                        ObaraSaikaRecursion.electron_repulsion_integral(ga_1, g_b, g_c, g_d, m + 1)
                    )
                    integral += angular_a1[i] / (2 * p) * (
                            ObaraSaikaRecursion.electron_repulsion_integral(ga_2, g_b, g_c, g_d, m) - (
                            q / (p + q) * ObaraSaikaRecursion.electron_repulsion_integral(ga_2, g_b, g_c, g_d, m + 1)
                    )
                    )
                    integral += g_c.angular[i] / (2 * (p + q)) * (
                        ObaraSaikaRecursion.electron_repulsion_integral(ga_1, g_b, gc_1, g_d, m + 1)
                    )
                    # print(f"[a({g_a.angular}) b({g_b.angular}) | c({g_c.angular}) d({g_d.angular})]{m}: {integral}")
                    return integral

            elif any(c > 0 for c in g_c.angular):
                return ObaraSaikaRecursion.electron_repulsion_integral(
                    g_c, g_d, g_a, g_b, m
                )

        elif any(b > 0 for b in g_b.angular):

            for cord in range(3):
                if g_b.angular[cord] == 0:
                    continue
                deltal = tuple(1 if idx == cord else 0 for idx in range(3))
                dist_cord = (g_a.center - g_b.center)[cord]
                angular_a1 = tuple(a + l for a, l in zip(g_a.angular, deltal))
                angular_b1 = tuple(b - l for b, l in zip(g_b.angular, deltal))
                g_a1 = PrimitiveGaussian(
                    g_a.exponent, g_a.center, angular_a1
                )
                g_b1 = PrimitiveGaussian(
                    g_b.exponent, g_b.center, angular_b1
                )
                integral = ObaraSaikaRecursion.electron_repulsion_integral(
                    g_a1, g_b1, g_c, g_d, m
                )
                integral += dist_cord * ObaraSaikaRecursion.electron_repulsion_integral(
                    g_a, g_b1, g_c, g_d, m
                )
                return integral

        elif any(d > 0 for d in g_d.angular):
            return ObaraSaikaRecursion.electron_repulsion_integral(
                g_c, g_d, g_a, g_b, m
            )




if __name__ == "__main__":
    ga = PrimitiveGaussian(
        0.5, np.array([0, 0, 0.5]), (1, 0, 0)
    )
    gb = PrimitiveGaussian(
        0.5, np.array([0, 0.5, 0]), (0, 1, 0)
    )
    gc = PrimitiveGaussian(
        0.5, np.array([0, 1, 0]), (0, 0, 1)
    )
    gd = PrimitiveGaussian(
        0.5, np.array([1, 0, 0]), (0, 0, 0)
    )
    int = ObaraSaikaRecursion.electron_repulsion_integral(
        ga, gb, gc, gd, 0
    )
    print(int)