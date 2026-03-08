import numpy as np
from typing import List, Dict, Tuple

from Molecules import Atom, Molecule
from BasisSet import BasisSetManager
from Gaussian import BasisFunction
from Integrals import OneElectronIntegrals, TwoElectronIntegrals
from Cart2Sph import CartToSph
from Grid import MoleculeGrid

np.set_printoptions(
    threshold=200,
    linewidth=200,
    precision=4
)


# ==================== Hartree-Fock Calculator ====================
class B3LYP:
    """
    Main class for performing closed‑shell Hartree–Fock calculations.
    Handles basis set construction, integral evaluation, SCF iterations,
    and extraction of molecular orbital energies.
    """

    def __init__(self, molecule: Molecule, basis_name: str = 'STO-3G'):
        """
        Initialise the Hartree–Fock calculator.

        Parameters
        ----------
        molecule : Molecule
            The molecular system to be studied.
        basis_name : str
            Name of the basis set (e.g., 'STO-3G', '3-21g').
        """
        self.molecule = molecule
        self.basis_name = basis_name
        self.basis_set_manager = BasisSetManager(basis_name)

        # Build the list of primitive Cartesian Gaussian basis functions
        self.basis_functions = self._construct_basis_functions()
        self._print_basis_functions()  # Optional: prints details of each basis function
        self.nbasis = len(self.basis_functions)  # Number of Cartesian Gaussian basis functions
        self.nocc = molecule.nelectrons // 2  # Number of occupied orbitals (closed‑shell)

        # Build transformation matrix from Cartesian to spherical harmonic Gaussians
        self.X = CartToSph(self.basis_functions).build_transform_matrix()  # Transformation matrix, shape (n_sph, nbasis)
        self.n_sph = self.X.shape[0]  # Number of spherical harmonic basis functions

        # Initialise matrices (all will be in the spherical harmonic basis)
        self.S = np.zeros((self.n_sph, self.n_sph))  # Overlap matrix
        self.T = np.zeros((self.n_sph, self.n_sph))  # Kinetic energy matrix
        self.V = np.zeros((self.n_sph, self.n_sph))  # Nuclear attraction matrix
        self.H = np.zeros((self.n_sph, self.n_sph))  # Core Hamiltonian (T + V)
        self.J = np.zeros((self.n_sph, self.n_sph))  # Coulomb matrix
        self.P = np.zeros((self.n_sph, self.n_sph))  # Density matrix
        self.C = np.zeros((self.n_sph, self.n_sph))  # Molecular orbital coefficients
        self.KS = np.zeros((self.n_sph, self.n_sph))  # KS matrix
        self.V_xc = np.zeros((self.n_sph, self.n_sph))  # Exchange-Correlation matrix

        self.epsilon = np.zeros(self.n_sph)  # Orbital energies

        # Electron repulsion integrals (ERI) in Cartesian basis (4‑index array)
        self.ERI = np.zeros([self.nbasis, self.nbasis, self.nbasis, self.nbasis])
        # ERI in spherical harmonic basis
        self.SphERI = np.zeros([self.n_sph, self.n_sph, self.n_sph, self.n_sph])

        # Compute one‑electron integrals and transform to spherical harmonic basis
        self._compute_one_electron_integrals()
        # Compute two‑electron integrals and transform
        self._compute_two_electron_integrals()

        # molecule grid for DFT calculations
        self.mol_grid = MoleculeGrid(molecule, n_radial=75, preset="fine").build_molecule_grid()
        self.phi_sph = self._compute_spherical_basis_value()
        self.v_xc_grid = np.zeros(len(self.mol_grid.points))

    def _construct_basis_functions(self) -> List[BasisFunction]:
        """
        Build a list of Cartesian Gaussian basis functions for all atoms in the molecule.
        Each function corresponds to a specific contraction (primitive exponents and coefficients)
        and a specific angular momentum component.
        """
        basis_functions = []
        for atom in self.molecule.atoms:
            # Get the basis set information for this atom (shells with exponents, coefficients, angular momenta)
            shells = self.basis_set_manager.get_basis_for_atom(atom.symbol)
            for shell in shells:
                angular_list = shell["angular"]
                for angular in angular_list:
                    # Create one basis function per angular momentum component
                    basis_func = BasisFunction(
                        center=atom.position,
                        coefficients=shell["coefficients"],
                        exponents=shell["exponents"],
                        angular=angular
                    )
                    basis_functions.append(basis_func)

        return basis_functions

    def _print_basis_functions(self):
        """Print a summary of all Cartesian basis functions (useful for debugging)."""
        for func in self.basis_functions:
            print(f"center: {func.center}, angular: {func.angular}, "
                  f"coefficients: {func.coefficients}, exponents: {func.exponents}")

    def _compute_one_electron_integrals(self):
        """
        Compute overlap, kinetic, and nuclear attraction integrals in the Cartesian basis,
        then transform them to the spherical harmonic basis using the transformation matrix X.
        The core Hamiltonian H = T + V is also formed.
        """
        nuclei = self.molecule.nuclei
        # Matrices in Cartesian basis
        S_cart = np.zeros([self.nbasis, self.nbasis])
        T_cart = np.zeros([self.nbasis, self.nbasis])
        V_cart = np.zeros([self.nbasis, self.nbasis])

        # Loop over all pairs of Cartesian basis functions
        # (only upper triangle computed, symmetry enforced)
        for i in range(self.nbasis):
            for j in range(i, self.nbasis):
                # Overlap integral S_ij = ∫ φ_i(r) φ_j(r) dr
                S_cart[i, j] = S_cart[j, i] = OneElectronIntegrals.overlap_integral(
                    self.basis_functions[i], self.basis_functions[j]
                )

                # Kinetic energy integral T_ij = -½ ∫ φ_i(r) ∇² φ_j(r) dr
                T_cart[i, j] = T_cart[j, i] = OneElectronIntegrals.kinetic_energy(
                    self.basis_functions[i], self.basis_functions[j]
                )

                # Nuclear attraction integral V_ij = -∑_A ∫ φ_i(r) (Z_A / |r - R_A|) φ_j(r) dr
                V_cart[i, j] = V_cart[j, i] = OneElectronIntegrals.nuclear_attraction(
                    self.basis_functions[i], self.basis_functions[j], nuclei
                )

        # Transform to spherical harmonic basis: A_sph = X @ A_cart @ X.T
        X = self.X
        self.S = X @ S_cart @ X.T
        self.T = X @ T_cart @ X.T
        self.V = X @ V_cart @ X.T
        # Core Hamiltonian
        self.H = self.T + self.V

    def _compute_two_electron_integrals(self):
        """
        Compute the two‑electron repulsion integrals (ERI) in the Cartesian basis,
        then transform them to the spherical harmonic basis using the transformation matrix X.
        The integrals are stored in the four‑index array self.SphERI.
        """
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
                        # Store all symmetry‑equivalent permutations
                        self.ERI[i, j, k, l] = self.ERI[j, i, k, l] = self.ERI[i, j, l, k] = self.ERI[j, i, l, k] = integral
                        self.ERI[k, l, i, j] = self.ERI[l, k, i, j] = self.ERI[k, l, j, i] = self.ERI[l, k, j, i] = integral

        # Transform the ket indices (k,l) to spherical harmonic basis
        for i in range(self.nbasis):
            for j in range(self.nbasis):
                cart_eri = self.ERI[i, j, :, :]
                half_sph_eri[i, j, :, :] = X @ cart_eri @ X.T

        # Transform the bra indices (i,j) to spherical harmonic basis
        for k in range(n_sph):
            for l in range(n_sph):
                cart_eri = half_sph_eri[:, :, k, l]
                sph_eri[:, :, k, l] = X @ cart_eri @ X.T
        self.SphERI = sph_eri

        # An alternative, more compact transformation using einsum would be:
        # self.ERI = np.einsum('pqrs,pi,qj,rk,sl->ijkl', self.ERI, X, X, X, X)

    def _compute_spherical_basis_value(self):
        points = self.mol_grid.points

        phi_cart = np.zeros([points.shape[0], len(self.basis_functions)])
        for idx, func in enumerate(self.basis_functions):
            phi_cart[:, idx] = np.array([func.value(r) for r in points])

        phi_sph = phi_cart @ self.X.T

        return phi_sph

    def _compute_electron_density(self):

        return np.sum((self.phi_sph @ self.P) * self.phi_sph, axis=1)

    def _compute_LDA_xc(self):

        # 常数定义
        C_x = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0)  # ≈ -0.7385588

        rho = self._compute_electron_density()
        # 初始化
        n_points = len(rho)
        e_x = np.zeros(n_points)
        e_c = np.zeros(n_points)
        v_x = np.zeros(n_points)
        v_c = np.zeros(n_points)

        for g in range(n_points):
            rho_g = rho[g]

            # 避免除零或负数
            if rho_g <= 1e-12:
                e_x[g] = 0.0
                e_c[g] = 0.0
                v_x[g] = 0.0
                v_c[g] = 0.0
                continue

            # --- 交换部分 (Slater) ---
            # 能量密度 ε_x = C_x * ρ^(1/3)
            rho_13 = rho_g ** (1.0 / 3.0)
            e_x[g] = C_x * rho_13

            # 势函数 v_x = (4/3) * C_x * ρ^(1/3) = (4/3) * e_x
            v_x[g] = (4.0 / 3.0) * e_x[g]

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
        e_xc_density = rho * (e_x + e_c)  # 能量密度 (每体积)
        E_xc = np.sum(self.mol_grid.weights * e_xc_density)

        # 总交换相关势
        v_xc_total = v_x + v_c

        return v_xc_total, E_xc

    def _compute_kohn_sham_matrix(self):
        """
        Build the Fock matrix in the spherical harmonic basis from the current density matrix.
        F = H + J + V, where J and K are the Coulomb and exchange matrices respectively.
        """
        self.J = np.zeros([self.n_sph, self.n_sph])

        # Sum over all four indices of the two‑electron integrals
        self.J = np.einsum('kl,ijkl->ij', self.P, self.SphERI)

        self.V_xc = self.phi_sph.T @ (np.tile((self.v_xc_grid * self.mol_grid.weights).reshape(-1, 1), (1, self.n_sph)) * self.phi_sph)

        self.KS = self.H + self.J + self.V_xc

    def scf_iteration(self, max_iter: int = 100, tol: float = 1e-8) -> Dict:
        """
        Perform self‑consistent field (SCF) iterations until convergence or max_iter.

        Parameters
        ----------
        max_iter : int
            Maximum number of SCF cycles.
        tol : float
            Convergence threshold for the change in total energy between iterations.

        Returns
        -------
        dict
            Dictionary containing the final energy, convergence status, and number of iterations.
        """
        print("Starting Hartree–Fock SCF calculation")
        print(f"Molecule: {len(self.molecule.atoms)} atoms")
        print(f"Basis set: {self.basis_name}")
        print(f"Cartesian Gaussian functions: {self.nbasis}")
        print(f"Spherical harmonic functions: {self.n_sph}")
        print(f"Number of occupied orbitals: {self.nocc}")

        # Initial guess: diagonalise the core Hamiltonian
        print("\nInitial guess...")
        eigenvalues, eigenvectors = self._solve_roothaan_equation(self.H, self.S)
        self.C = eigenvectors
        self.epsilon = eigenvalues

        # Build the initial density matrix from the occupied MOs
        self._build_density_matrix()

        # compute initial energy, e_xc and v_xc
        energy_old = self._compute_total_energy()
        for iteration in range(max_iter):

            # 1.Compute Fock matrix from current density
            self._compute_kohn_sham_matrix()

            # 2.Solve the Roothaan–Hall equations: F C = S C ε
            eigenvalues, eigenvectors = self._solve_roothaan_equation(self.KS, self.S)
            self.C = eigenvectors
            self.epsilon = eigenvalues

            # 3.Update density matrix
            self._build_density_matrix()

            # 4.Compute total electronic + nuclear energy
            energy = self._compute_total_energy()

            # 5.Check convergence
            delta_energy = abs(energy - energy_old)
            print(f"Iteration {iteration + 1:3d}: total energy = {energy:15.10f} Hartree, "
                  f"ΔE = {delta_energy:10.3e}")

            if delta_energy < tol:
                print(f"\nSCF converged in {iteration + 1} iterations")
                print(f"Final total energy: {energy:.10f} Hartree")
                print(f"Electronic energy: {energy - self.molecule.nuclear_repulsion_energy:.10f} Hartree")
                print(f"Nuclear repulsion energy: {self.molecule.nuclear_repulsion_energy:.10f} Hartree")
                return {
                    "energy": energy,
                    "converged": True,
                    "iterations": iteration + 1
                }

            energy_old = energy

        print("Warning: SCF did not converge!")
        return {
            "energy": energy_old,
            "converged": False,
            "iterations": max_iter
        }

    def _solve_roothaan_equation(self, F: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the generalised eigenvalue problem F C = S C ε.
        Uses Cholesky decomposition of S to transform to an orthonormal basis.

        Parameters
        ----------
        F : ndarray
            Fock (or core Hamiltonian) matrix.
        S : ndarray
            Overlap matrix.

        Returns
        -------
        epsilon : ndarray
            Eigenvalues (orbital energies).
        C : ndarray
            Eigenvectors (MO coefficients) in the original (non‑orthogonal) basis.
        """
        # 1.Cholesky factorisation: S = L L^T
        L = np.linalg.cholesky(S)
        L_inv = np.linalg.inv(L)

        # 2.Transform F to the orthogonal basis: F' = L^{-1} F L^{-T}
        F_prime = L_inv @ F @ L_inv.T

        # 3.Solve standard eigenvalue problem: F' C' = C' ε
        epsilon_prime, C_prime = np.linalg.eigh(F_prime)

        # 4.Back‑transform eigenvectors to original basis: C = L^{-T} C'
        C = L_inv.T @ C_prime

        return epsilon_prime, C

    def _build_density_matrix(self):
        """
        Construct the density matrix from the current MO coefficients.
        For a closed‑shell system: P = 2 * C_occ C_occ^T,
        where C_occ contains the occupied orbitals.
        """
        C_occ = self.C[:, :self.nocc]
        self.P = 0.6 * C_occ @ C_occ.T + 0.7 * self.P

    def _compute_total_energy(self) -> float:
        """
        Compute the total Hartree–Fock energy:
        E_total = ½ Tr[P (H + F)] + E_nuc
        """
        # 单电子部分：Tr[P * H_core]
        E_one = np.einsum('ij,ji->', self.P, self.H)  # 或 np.sum(P * H_core)
        self.J = np.einsum('kl,ijkl->ij', self.P, self.SphERI)
        E_j = 0.5 * np.einsum('ij,ji->', self.P, self.J)  # 或 0.5 * np.sum(P * J)
        self.v_xc_grid, self.E_xc = self._compute_LDA_xc()
        # 总电子能量
        E_elec = E_one + E_j + self.E_xc + self.molecule.nuclear_repulsion_energy
        return E_elec



# ==================== Test code ====================
if __name__ == "__main__":
    # Create a water molecule with given coordinates in angstroms （same with Gaussian）
    atom1 = Atom('O', np.array([-0.05591054, 2.17252383, 0.00000000]))
    atom2 = Atom('H', np.array([0.90408946, 2.17252383, 0.00000000]))
    atom3 = Atom('H', np.array([-0.37636513, 3.07745966, 0.00000000]))

    h2o_molecule = Molecule([atom1, atom2, atom3], charge=0, multiplicity=1)

    # Run a Hartree–Fock calculation using the 3-21G basis set
    hf_calculator = B3LYP(h2o_molecule, basis_name='6-31G')
    result = hf_calculator.scf_iteration(max_iter=500, tol=1e-6)


    # Print final molecular orbital energies
    print("\nMolecular orbital energies (Hartree):")
    for i, eps in enumerate(hf_calculator.epsilon):
        print(f"Orbital {i + 1:3d}: {eps:12.6f}")

