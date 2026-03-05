import numpy as np
from typing import List, Dict, Tuple

from Molecules import Atom, Molecule
from BasisSet import BasisSetManager
from Gaussian import BasisFunction
from Integrals import OneElectronIntegrals, TwoElectronIntegrals
from Cart2Sph import CartToSph

np.set_printoptions(
    threshold=200,
    linewidth=200,
    precision=4
)


class UnrestrictedHF:
    def __init__(self, molecule: Molecule, basis_name: str = "STO-3G"):
        self.molecule = molecule
        self.basis_name = basis_name
        self.basis_set_manager = BasisSetManager(basis_name)

        self.multiplicity = molecule.multiplicity
        self.n_e = molecule.nelectrons
        self.n_alpha = (self.n_e + self.multiplicity - 1) // 2
        self.n_beta = self.n_e - self.n_alpha

        self.basis_functions = self._construct_basis_functions()
        self.n_basis = len(self.basis_functions)

        # 计算笛卡尔高斯到球谐高斯的变换矩阵
        cart2sph = CartToSph(self.basis_functions)
        self.X = cart2sph.build_transform_matrix()
        self.n_sph = self.X.shape[0]

        # 初始化矩阵
        self.S = np.zeros((self.n_sph, self.n_sph))  # 重叠矩阵
        self.T = np.zeros((self.n_sph, self.n_sph))  # 动能矩阵
        self.V = np.zeros((self.n_sph, self.n_sph))  # 核吸引矩阵
        self.H = np.zeros((self.n_sph, self.n_sph))  # 核心哈密顿矩阵

        self.F_alpha = np.zeros((self.n_sph, self.n_sph))
        self.F_beta = np.zeros((self.n_sph, self.n_sph)) # Fock矩阵
        self.P_alpha = np.zeros((self.n_sph, self.n_sph))
        self.P_beta = np.zeros((self.n_sph, self.n_sph))  # 密度矩阵
        self.C_alpha = np.zeros((self.n_sph, self.n_sph))
        self.C_beta = np.zeros((self.n_sph, self.n_sph))  # 分子轨道系数

        # 轨道能量
        self.epsilon_alpha = np.zeros([self.n_sph])
        self.epsilon_beta = np.zeros([self.n_sph])

        # 笛卡尔双电子积分
        self.ERI = np.zeros([self.n_basis, self.n_basis, self.n_basis, self.n_basis])
        # 球谐高斯双电子积分
        self.SphERI = np.zeros([self.n_sph, self.n_sph, self.n_sph, self.n_sph])
        # 计算单电子积分
        self._compute_one_electron_integrals()
        # 计算双电子积分
        self._compute_two_electron_integrals()

    def _construct_basis_functions(self) -> List[BasisFunction]:
        """构建笛卡尔高斯基函数列表"""
        basis_functions = []

        for atom in self.molecule.atoms:
            # 获取原子的基组信息
            shells = self.basis_set_manager.get_basis_for_atom(atom.symbol)

            for shell in shells:

                angular_list = shell["angular"]

                for angular in angular_list:
                    basis_func = BasisFunction(
                        center=atom.position,
                        coefficients=shell["coefficients"],
                        exponents=shell["exponents"],
                        angular=angular
                    )
                    basis_functions.append(basis_func)

        return basis_functions

    def _compute_one_electron_integrals(self):
        """计算单电子积分"""
        nuclei = self.molecule.nuclei
        S_cart = np.zeros([self.n_basis, self.n_basis])  # 笛卡尔重叠矩阵
        T_cart = np.zeros([self.n_basis, self.n_basis])  # 笛卡尔动能矩阵
        V_cart = np.zeros([self.n_basis, self.n_basis])  # 笛卡尔核吸引矩阵

        for i in range(self.n_basis):
            for j in range(i, self.n_basis):
                # 重叠积分
                S_cart[i, j] = S_cart[j, i] = OneElectronIntegrals.overlap_integral(
                    self.basis_functions[i], self.basis_functions[j]
                )

                # 动能积分
                T_cart[i, j] = T_cart[j, i] = OneElectronIntegrals.kinetic_energy(
                    self.basis_functions[i], self.basis_functions[j]
                )

                # 核吸引积分
                V_cart[i, j] = V_cart[j, i] = OneElectronIntegrals.nuclear_attraction(
                    self.basis_functions[i], self.basis_functions[j], nuclei
                )

        # 核心哈密顿矩阵
        X = self.X
        self.S = X @ S_cart @ X.T
        self.T = X @ T_cart @ X.T
        self.V = X @ V_cart @ X.T

        self.H = self.T + self.V

    def _compute_two_electron_integrals(self):
        """计算双电子积分并存储在四维数组中"""
        X = self.X
        n_sph = X.shape[0]
        sph_eri = np.zeros([n_sph, n_sph, n_sph, n_sph])
        half_sph_eri = np.zeros([self.n_basis, self.n_basis, n_sph, n_sph])
        for i in range(self.n_basis):
            for j in range(i, self.n_basis):
                for k in range(self.n_basis):
                    for l in range(k, self.n_basis):
                        integral = TwoElectronIntegrals.electron_repulsion(
                            self.basis_functions[i], self.basis_functions[j],
                            self.basis_functions[k], self.basis_functions[l]
                        )
                        self.ERI[i, j, k, l] = self.ERI[j, i, k, l] = self.ERI[i, j, l, k] = self.ERI[j, i, l, k] = integral
                        self.ERI[k, l, i, j] = self.ERI[l, k, i, j] = self.ERI[k, l, j, i] = self.ERI[l, k, j, i] = integral

        # 将ket部分转化为球谐高斯基函数
        for i in range(self.n_basis):
            for j in range(self.n_basis):
                cart_eri = self.ERI[i, j, :, :]
                half_sph_eri[i, j, :, :] = X @ cart_eri @ X.T
        for k in range(n_sph):
            for l in range(n_sph):
                cart_eri = half_sph_eri[:, :, k, l]
                sph_eri[:, :, k, l] = X @ cart_eri @ X.T
        self.SphERI = sph_eri

        #self.ERI = np.einsum('pqrs,pi,qj,rk,sl->ijkl', self.ERI, X, X, X, X)

    def _compute_fock_matrix(self):
        """计算Fock矩阵"""
        # 初始化电子排斥矩阵
        J = np.zeros([self.n_sph, self.n_sph])
        K_alpha = np.zeros([self.n_sph, self.n_sph])
        K_beta = np.zeros([self.n_sph, self.n_sph])

        # 计算双电子积分并构建J和K矩阵
        for i in range(self.n_sph):
            for j in range(self.n_sph):
                for k in range(self.n_sph):
                    for l in range(self.n_sph):

                        # 库仑和交换贡献
                        J[i, j] += self.P_alpha[k, l] * self.SphERI[i, j, k, l] + (
                            self.P_beta[k, l] * self.SphERI[i, j, k, l]
                        )
                        K_alpha[i, j] += self.P_alpha[k, l] * self.SphERI[i, k, j, l]
                        K_beta[i, j] += self.P_beta[k, l] * self.SphERI[i, k, j, l]

        # Fock矩阵
        self.F_alpha = self.H + J - K_alpha
        self.F_beta = self.H + J - K_beta

    @staticmethod
    def _solve_roothaan_equation(F: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """求解Roothaan方程 F·C = S·C·ε"""
        # 对S进行Cholesky分解
        L = np.linalg.cholesky(S)
        L_inv = np.linalg.inv(L)

        # 变换到正交基
        F_prime = L_inv @ F @ L_inv.T

        # 求解标准本征值问题
        epsilon_prime, C_prime = np.linalg.eigh(F_prime)

        # 变换回原基组
        C = L_inv.T @ C_prime

        return epsilon_prime, C

    def _build_density_matrix(self):
        """构建密度矩阵"""
        # 占据轨道系数
        C_oc_alpha = self.C_alpha[:, :self.n_alpha]
        C_oc_beta = self.C_beta[:, :self.n_beta]

        # 密度矩阵
        print("密度矩阵")
        print(self.P_alpha)
        self.P_alpha = C_oc_alpha @ C_oc_alpha.T
        self.P_beta = C_oc_beta @ C_oc_beta.T

    def _compute_total_energy(self):

        e1 = np.trace(self.P_alpha @ (self.F_alpha + self.H))
        e2 = np.trace(self.P_beta @ (self.F_beta + self.H))

        return 0.5 * (e1 + e2) + self.molecule.nuclear_repulsion_energy

    def scf_iteration(self, max_iter: int = 100, tol: float = 1e-8) -> Dict:
        """SCF迭代"""
        print("开始Hartree-Fock SCF计算")
        print(f"分子: {len(self.molecule.atoms)} 个原子")
        print(f"基组: {self.basis_name}")
        print(f"笛卡尔高斯基函数: {self.n_basis}")
        print(f"球谐高斯基函数: {self.n_sph}")
        print(f"占据alpha轨道数: {self.n_alpha}")
        print(f"占据beta轨道数: {self.n_beta}")

        # 初始猜测：对角化核心哈密顿矩阵
        print("\n初始猜测...")
        eigenvalues, eigenvectors = self._solve_roothaan_equation(self.H, self.S)
        self.C_alpha = eigenvectors
        self.C_beta = eigenvectors
        self.epsilon_alpha = eigenvalues
        self.epsilon_beta = eigenvalues

        # 构建初始密度矩阵
        self._build_density_matrix()

        energy_old = 0.0
        for iteration in range(max_iter):
            # 计算Fock矩阵
            self._compute_fock_matrix()

            # 求解Roothaan方程
            alpha_eigenvalues, alpha_eigenvectors = self._solve_roothaan_equation(self.F_alpha, self.S)
            self.C_alpha = alpha_eigenvectors
            self.epsilon_alpha = alpha_eigenvalues
            beta_eigenvalues, beta_eigenvectors = self._solve_roothaan_equation(self.F_beta, self.S)
            self.C_beta = beta_eigenvectors
            self.epsilon_beta = beta_eigenvalues

            # 构建新密度矩阵
            self._build_density_matrix()

            # 计算总能量
            energy = self._compute_total_energy()

            # 检查收敛
            delta_energy = abs(energy - energy_old)
            print(f"迭代 {iteration + 1:3d}: 总能量 = {energy:15.10f}, ΔE = {delta_energy:10.3e}")

            if delta_energy < tol:
                print(f"\nSCF收敛于 {iteration + 1} 次迭代")
                print(f"最终总能量: {energy:.10f} Hartree")
                print(f"电子能量: {energy - self.molecule.nuclear_repulsion_energy:.10f} Hartree")
                print(f"核排斥能: {self.molecule.nuclear_repulsion_energy:.10f} Hartree")
                return {
                    "energy": energy,
                    "converged": True,
                    "iterations": iteration + 1
                }

            energy_old = energy

        print("警告：SCF未收敛！")
        return {
            "energy": energy_old,
            "converged": False,
            "iterations": max_iter
        }

    def compute_squared_S(self):

        s = (self.n_alpha - self.n_beta) // 2
        squared_S_exact = s * (s + 1)

        c_alpha = self.C_alpha[:, : self.n_alpha]
        print(c_alpha.shape)
        c_beta = self.C_beta[:, : self.n_beta]
        S_mo = c_alpha.T @ self.S @ c_beta

        squared_S_uhf = squared_S_exact + self.n_beta - np.sum(S_mo ** 2)

        return squared_S_uhf


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 创建水分子
    atom1 = Atom('O', np.array([-0.99041532, 0.74281145, 0.00000000]))
    atom2 = Atom('O', np.array([-2.15201532, 0.74281145, 0.00000000]))

    h2o_molecule = Molecule([atom1, atom2], charge=0, multiplicity=3)

    # Hartree-Fock计算
    hf_calculator = UnrestrictedHF(h2o_molecule, basis_name='3-21g')
    result = hf_calculator.scf_iteration(max_iter=100, tol=1e-6)

    # 输出分子轨道信息
    print("\nalpha分子轨道能量 (Hartree):")
    for i, eps in enumerate(hf_calculator.epsilon_alpha):
        print(f"轨道 {i + 1:3d}: {eps:12.6f}")
    print("\nbeta分子轨道能量 (Hartree):")
    for i, eps in enumerate(hf_calculator.epsilon_beta):
        print(f"轨道 {i + 1:3d}: {eps:12.6f}")

    S2 = hf_calculator.compute_squared_S()
    print(S2)

