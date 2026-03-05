import numpy as np
from typing import List, Dict, Tuple

from Molecules import Atom, Molecule
from BasisSet import BasisSetManager
from Gaussian import BasisFunction
from Integrals import OneElectronIntegrals, TwoElectronIntegrals
from Cart2Sph import CartToSph

np.set_printoptions(
    threshold=np.inf,
    linewidth=np.inf,
    precision=4
)   # 或 linewidth=200


# ==================== Hartree-Fock Calculator ====================
class HartreeFock:
    """Hartree-Fock计算主类"""

    def __init__(self, molecule: Molecule, basis_name: str = 'STO-3G'):
        self.molecule = molecule
        self.basis_name = basis_name
        self.basis_set_manager = BasisSetManager(basis_name)

        # 构建基函数列表
        self.basis_functions = self._construct_basis_functions()
        self._print_basis_functions()
        self.nbasis = len(self.basis_functions)
        self.nocc = molecule.nelectrons // 2  # 占据轨道数（闭壳层）

        # 计算笛卡尔高斯到球谐高斯的变换矩阵
        cart2sph = CartToSph(self.basis_functions)
        self.X = cart2sph.build_transform_matrix()
        self.n_sph = self.X.shape[0]

        # 初始化矩阵
        self.S = np.zeros((self.n_sph, self.n_sph))  # 重叠矩阵
        self.T = np.zeros((self.n_sph, self.n_sph))  # 动能矩阵
        self.V = np.zeros((self.n_sph, self.n_sph))  # 核吸引矩阵
        self.H = np.zeros((self.n_sph, self.n_sph))  # 核心哈密顿矩阵
        self.F = np.zeros((self.n_sph, self.n_sph))  # Fock矩阵
        self.P = np.zeros((self.n_sph, self.n_sph))  # 密度矩阵
        self.C = np.zeros((self.n_sph, self.n_sph))  # 分子轨道系数

        # 轨道能量
        self.epsilon = np.zeros([self.n_sph])

        # 笛卡尔双电子积分
        self.ERI = np.zeros([self.nbasis, self.nbasis, self.nbasis, self.nbasis])
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

    def _print_basis_functions(self):
        for func in self.basis_functions:
            print(f"中心: {func.center}, 角动量: {func.angular}, 系数: {func.coefficients}, 指数: {func.exponents}")

    def _compute_one_electron_integrals(self):
        """计算单电子积分"""
        nuclei = self.molecule.nuclei
        S_cart = np.zeros([self.nbasis, self.nbasis])  # 笛卡尔重叠矩阵
        T_cart = np.zeros([self.nbasis, self.nbasis])  # 笛卡尔动能矩阵
        V_cart = np.zeros([self.nbasis, self.nbasis])  # 笛卡尔核吸引矩阵

        for i in range(self.nbasis):
            for j in range(i, self.nbasis):
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
        print("S")
        print(self.S)
        print("T")
        print(self.T)
        print("V")
        print(self.V)

    def _compute_two_electron_integrals(self):
        """计算双电子积分并存储在四维数组中"""
        X = self.X
        n_sph = X.shape[0]
        sph_eri = np.zeros([n_sph, n_sph, n_sph, n_sph])
        half_sph_eri = np.zeros([self.nbasis, self.nbasis, n_sph, n_sph])
        for i in range(self.nbasis):
            for j in range(i, self.nbasis):
                for k in range(self.nbasis):
                    for l in range(k, self.nbasis):
                        integral = TwoElectronIntegrals.electron_repulsion(
                            self.basis_functions[i], self.basis_functions[j],
                            self.basis_functions[k], self.basis_functions[l]
                        )
                        self.ERI[i, j, k, l] = self.ERI[j, i, k, l] = self.ERI[i, j, l, k] = self.ERI[j, i, l, k] = integral
                        self.ERI[k, l, i, j] = self.ERI[l, k, i, j] = self.ERI[k, l, j, i] = self.ERI[l, k, j, i] = integral

        # 将ket部分转化为球谐高斯基函数
        for i in range(self.nbasis):
            for j in range(self.nbasis):
                cart_eri = self.ERI[i, j, :, :]
                half_sph_eri[i, j, :, :] = X @ cart_eri @ X.T
        for k in range(n_sph):
            for l in range(n_sph):
                cart_eri = half_sph_eri[:, :, k, l]
                sph_eri[:, :, k, l] = X @ cart_eri @ X.T
        self.SphERI = sph_eri

        #self.ERI = np.einsum('pqrs,pi,qj,rk,sl->ijkl', self.ERI, X, X, X, X)

    def compute_fock_matrix(self):
        """计算Fock矩阵"""
        # 初始化电子排斥矩阵
        J = np.zeros([self.n_sph, self.n_sph])
        K = np.zeros([self.n_sph, self.n_sph])

        # 计算双电子积分并构建J和K矩阵
        for i in range(self.n_sph):
            for j in range(self.n_sph):
                for k in range(self.n_sph):
                    for l in range(self.n_sph):

                        # 库仑和交换贡献
                        J[i, j] += self.P[k, l] * self.SphERI[i, j, k, l]
                        K[i, j] += 0.5 * self.P[k, l] * self.SphERI[i, k, j, l]

        # Fock矩阵
        self.F = self.H + J - K

    def scf_iteration(self, max_iter: int = 100, tol: float = 1e-8) -> Dict:
        """SCF迭代"""
        print("开始Hartree-Fock SCF计算")
        print(f"分子: {len(self.molecule.atoms)} 个原子")
        print(f"基组: {self.basis_name}")
        print(f"笛卡尔高斯基函数: {self.nbasis}")
        print(f"球谐高斯基函数: {self.n_sph}")
        print(f"占据轨道数: {self.nocc}")

        # 初始猜测：对角化核心哈密顿矩阵
        print("\n初始猜测...")
        eigenvalues, eigenvectors = self._solve_roothaan_equation(self.H, self.S)
        self.C = eigenvectors
        self.epsilon = eigenvalues

        # 构建初始密度矩阵
        self._build_density_matrix()

        energy_old = 0.0
        for iteration in range(max_iter):
            # 计算Fock矩阵
            self.compute_fock_matrix()

            # 求解Roothaan方程
            eigenvalues, eigenvectors = self._solve_roothaan_equation(self.F, self.S)
            self.C = eigenvectors
            self.epsilon = eigenvalues

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

    def _solve_roothaan_equation(self, F: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        C_occ = self.C[:, :self.nocc]

        # 密度矩阵
        self.P = 2.0 * C_occ @ C_occ.T

    def _compute_total_energy(self) -> float:
        """计算总能量"""
        # 电子能量
        electronic_energy = 0.5 * np.sum(self.P * (self.H + self.F))

        # 总能量
        total_energy = electronic_energy + self.molecule.nuclear_repulsion_energy

        return float(total_energy)


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 创建水分子
    atom1 = Atom('O', np.array([-0.05591054, 2.17252383, 0.00000000]))
    atom2 = Atom('H', np.array([0.90408946, 2.17252383, 0.00000000]))
    atom3 = Atom('H', np.array([-0.37636513, 3.07745966, 0.00000000]))

    h2o_molecule = Molecule([atom1, atom2, atom3], charge=0, multiplicity=1)

    # Hartree-Fock计算
    hf_calculator = HartreeFock(h2o_molecule, basis_name='3-21g')
    result = hf_calculator.scf_iteration(max_iter=50, tol=1e-6)

    # 输出分子轨道信息
    print("\n分子轨道能量 (Hartree):")
    for i, eps in enumerate(hf_calculator.epsilon):
        print(f"轨道 {i + 1:3d}: {eps:12.6f}")

