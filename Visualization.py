import numpy as np
import pyvista as pv
from typing import List

from UHF import UnrestrictedHF
from Molecules import Molecule, Atom
from Gaussian import BasisFunction
from Cart2Sph import CartToSph


class SpinDensityVisualizer:

    def __init__(self, mol: Molecule, P_spin: np.ndarray, basis_functions: List[BasisFunction]):

        self.molecule = mol
        self.basis_functions = basis_functions
        self.P_spin = P_spin
        self.x = None
        self.y = None
        self.z = None

    def spin_density_func(self, r: np.ndarray):

        n_basis = len(self.basis_functions)
        cart_matrix = np.zeros([n_basis, n_basis])
        for a in range(n_basis):
            for b in range(n_basis):
                cart_matrix[a, b] += self.basis_functions[a].value(r) * self.basis_functions[b].value(r)

        sph_matrix = self.X @ cart_matrix @ self.X.T
        spin_density = np.sum(self.P_spin * sph_matrix)
        return spin_density

    def build_grid(self, step: float = 0.1):
        positions = self.molecule.nuclei_array
        min_coord = np.min(positions, axis=0) - 5.0
        max_coord = np.max(positions, axis=0) + 5.0

        nx = int((max_coord[0] - min_coord[0]) / step) + 1
        ny = int((max_coord[1] - min_coord[1]) / step) + 1
        nz = int((max_coord[2] - min_coord[2]) / step) + 1

        self.x = np.linspace(min_coord[0], max_coord[0], nx)
        self.y = np.linspace(min_coord[1], max_coord[1], ny)
        self.z = np.linspace(min_coord[2], max_coord[2], nz)

        return self.x, self.y, self.z

    def compute_spin_density(self):

        x, y, z = self.build_grid()
        nx, ny, nz = len(x), len(y), len(z)

        points = np.array(np.meshgrid(x, y, z, indexing='ij')).reshape(3, -1).T

        phi_cart = np.zeros([points.shape[0], len(self.basis_functions)])
        for idx, func in enumerate(self.basis_functions):
            phi_cart[:, idx] = np.array([func.value(r) for r in points])

        cart2sph = CartToSph(self.basis_functions)
        X = cart2sph.build_transform_matrix()
        phi_sph = phi_cart @ X.T
        # 计算自旋密度：ρ_spin = ∑_{μν} P_spin[μν] φ_μ φ_ν = diag(phi_sph @ P_spin @ phi_sph.T)
        # 用公式：ρ_spin = sum( (phi_sph @ P_spin) * phi_sph, axis=1 )
        rho_spin_flat = np.sum((phi_sph @ self.P_spin) * phi_sph, axis=1)
        return rho_spin_flat.reshape(nx, ny, nz)

    def plot_spin_density(self, step: float = 0.1, threshold: float = 0.01):

        rho_spin = self.compute_spin_density()
        x, y, z = self.x, self.y, self.z

        # 创建 ImageData（均匀网格）
        grid = pv.ImageData()
        grid.origin = (x[0], y[0], z[0])
        grid.spacing = (step, step, step)
        grid.dimensions = (len(x), len(y), len(z))  # 点的数量

        # 2. 将自旋密度添加到网格
        grid.point_data["spin_density"] = rho_spin.flatten(order="F")

        # 3. 创建绘图对象
        plotter = pv.Plotter()

        # 4. 绘制正密度等值面（例如取值为最大值的20%作为阈值，或手动指定）
        positive_threshold = threshold  # 可根据数据范围调整
        positive = grid.contour([positive_threshold], scalars="spin_density", method='marching_cubes')
        plotter.add_mesh(positive, color="blue", opacity=0.7, label="Positive spin")

        # 5. 绘制负密度等值面

        negative_threshold = -threshold
        negative = grid.contour([negative_threshold], scalars="spin_density", method='marching_cubes')
        plotter.add_mesh(negative, color="red", opacity=0.7, label="Negative spin")

        # 6. 添加分子结构（原子位置）
        for atom in self.molecule.atoms:  # 假设 self.molecule 可用
            sphere = pv.Sphere(radius=0.5, center=atom.position)
            plotter.add_mesh(sphere, color="lightgray")

        # 7. 显示图例和坐标轴
        plotter.add_legend()
        plotter.show_grid()
        plotter.show()


if __name__ == "__main__":
    atom1 = Atom('O', np.array([-0.99041532, 0.74281145, 0.00000000]))
    atom2 = Atom('O', np.array([-2.15201532, 0.74281145, 0.00000000]))

    molecule = Molecule([atom1, atom2], charge=0, multiplicity=3)

    uhf = UnrestrictedHF(molecule, basis_name='sto-3g')

    basis_functions = uhf.basis_functions

    #result = uhf.scf_iteration()

    #np.save('pspin.npy', p_spin)
    p_spin = np.load('pspin.npy')

    spinviwer = SpinDensityVisualizer(uhf.molecule, p_spin, uhf.basis_functions)
    #spin_matrix = spinviwer.compute_spin_density()

    spinviwer.plot_spin_density()





















