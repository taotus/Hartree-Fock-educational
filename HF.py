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


# ==================== Hartree-Fock Calculator ====================
class HartreeFock:
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
        cart2sph = CartToSph(self.basis_functions)
        self.X = cart2sph.build_transform_matrix()  # Transformation matrix, shape (n_sph, nbasis)
        self.n_sph = self.X.shape[0]  # Number of spherical harmonic basis functions

        # Initialise matrices (all will be in the spherical harmonic basis)
        self.S = np.zeros((self.n_sph, self.n_sph))  # Overlap matrix
        self.T = np.zeros((self.n_sph, self.n_sph))  # Kinetic energy matrix
        self.V = np.zeros((self.n_sph, self.n_sph))  # Nuclear attraction matrix
        self.H = np.zeros((self.n_sph, self.n_sph))  # Core Hamiltonian (T + V)
        self.F = np.zeros((self.n_sph, self.n_sph))  # Fock matrix
        self.P = np.zeros((self.n_sph, self.n_sph))  # Density matrix
        self.C = np.zeros((self.n_sph, self.n_sph))  # Molecular orbital coefficients

        self.epsilon = np.zeros(self.n_sph)  # Orbital energies

        # Electron repulsion integrals (ERI) in Cartesian basis (4‑index array)
        self.ERI = np.zeros([self.nbasis, self.nbasis, self.nbasis, self.nbasis])
        # ERI in spherical harmonic basis
        self.SphERI = np.zeros([self.n_sph, self.n_sph, self.n_sph, self.n_sph])

        # Compute one‑electron integrals and transform to spherical harmonic basis
        self._compute_one_electron_integrals()
        # Compute two‑electron integrals and transform
        self._compute_two_electron_integrals()

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

    def compute_fock_matrix(self):
        """
        Build the Fock matrix in the spherical harmonic basis from the current density matrix.
        F = H + J - K, where J and K are the Coulomb and exchange matrices respectively.
        """
        J = np.zeros([self.n_sph, self.n_sph])
        K = np.zeros([self.n_sph, self.n_sph])

        # Sum over all four indices of the two‑electron integrals
        for i in range(self.n_sph):
            for j in range(self.n_sph):
                for k in range(self.n_sph):
                    for l in range(self.n_sph):
                        # Coulomb contribution: J_ij = ∑_kl P_kl (ij|kl)
                        J[i, j] += self.P[k, l] * self.SphERI[i, j, k, l]
                        # Exchange contribution: K_ij = ½ ∑_kl P_kl (ik|jl)
                        K[i, j] += 0.5 * self.P[k, l] * self.SphERI[i, k, j, l]

        self.F = self.H + J - K

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

        energy_old = 0.0
        for iteration in range(max_iter):

            # 1.Compute Fock matrix from current density
            self.compute_fock_matrix()

            # 2.Solve the Roothaan–Hall equations: F C = S C ε
            eigenvalues, eigenvectors = self._solve_roothaan_equation(self.F, self.S)
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
        self.P = 2.0 * C_occ @ C_occ.T

    def _compute_total_energy(self) -> float:
        """
        Compute the total Hartree–Fock energy:
        E_total = ½ Tr[P (H + F)] + E_nuc
        """
        electronic_energy = 0.5 * np.sum(self.P * (self.H + self.F))
        total_energy = electronic_energy + self.molecule.nuclear_repulsion_energy
        return float(total_energy)


# ==================== Test code ====================
if __name__ == "__main__":
    # Create a water molecule with given coordinates in angstroms （same with Gaussian）
    atom1 = Atom('O', np.array([-0.05591054, 2.17252383, 0.00000000]))
    atom2 = Atom('H', np.array([0.90408946, 2.17252383, 0.00000000]))
    atom3 = Atom('H', np.array([-0.37636513, 3.07745966, 0.00000000]))

    h2o_molecule = Molecule([atom1, atom2, atom3], charge=0, multiplicity=1)

    # Run a Hartree–Fock calculation using the 3-21G basis set
    hf_calculator = HartreeFock(h2o_molecule, basis_name='STO-3G')
    result = hf_calculator.scf_iteration(max_iter=50, tol=1e-6)

    # Print final molecular orbital energies
    print("\nMolecular orbital energies (Hartree):")
    for i, eps in enumerate(hf_calculator.epsilon):
        print(f"Orbital {i + 1:3d}: {eps:12.6f}")

