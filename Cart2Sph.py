import numpy as np
import scipy
from typing import List, Tuple

from Gaussian import BasisFunction
from BasisSet import BasisSetManager
from Molecules import Atom, Molecule


class CartToSph:
    """
    Class to construct the transformation matrix from Cartesian Gaussian basis functions
    to spherical harmonic Gaussian basis functions.

    For each angular momentum l, a fixed transformation matrix is provided (s, p, d).
    These matrices convert the (lx, ly, lz) Cartesian components into the standard
    real spherical harmonic combinations.
    """

    # Transformation matrix for s‑type functions (l=0)
    # One Cartesian s function maps to one spherical s function.
    s_matrix = np.array(
        [[1]]
    )

    # Transformation matrix for p‑type functions (l=1)
    # Three Cartesian p functions (px, py, pz) map to three spherical p functions.
    # The matrix is the identity because the Cartesian p functions already match
    # the solid spherical harmonics (px, py, pz) up to a normalization that is handled elsewhere.
    p_matrix = np.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]
    )
    # Transformation matrix for d‑type functions (l=2)
    # There are 6 Cartesian d functions (xx, yy, zz, xy, xz, yz) and
    # 5 spherical d functions (d_{z²}, d_{xz}, d_{yz}, d_{x²‑y²}, d_{xy}).
    # d_{x^2-y^2} = sqrt(3)/2 * (xx - yy)
    # d_{xz} = sqrt(3) * xz
    # d_{z^2} = 1/2 * (2*z^2 -x^2 - y^2)
    # d_{yz} = sqrt(3)/2 * yz
    # d_{xy} = sqrt(3)/2 * xy
    d_matrix = np.array(
        [[0, 0, 0, 0.8660254038, -0.8660254038, 0],
         [0, 1.732050808, 0, 0, 0, 0],
         [0, 0, 0, -0.5, -0.5, 1],
         [0, 0, 1.732050808, 0, 0, 0],
         [1.732050808, 0, 0, 0, 0, 0]]
    )

    def __init__(self, basis_functions: List[BasisFunction]):
        """
        Initialise the transformer with a list of Cartesian basis functions.

        Parameters
        ----------
        basis_functions : List[BasisFunction]
            The Cartesian Gaussian basis functions in the order they appear.
            It is assumed that functions are grouped by angular momentum on each atom.
        """
        self.basis_functions = basis_functions

    def _get_matrix(self, l) -> np.ndarray:
        """
        Return the transformation matrix for a given angular momentum quantum number l.

        Parameters
        ----------
        l : int
            Angular momentum (0 for s, 1 for p, 2 for d).

        Returns
        -------
        np.ndarray
            Transformation matrix of shape (n_sph, n_cart) for this l.
        """
        if l == 0:
            return self.s_matrix
        elif l == 1:
            return self.p_matrix
        elif l == 2:
            return self.d_matrix
        else:
            raise NotImplementedError(f"Transformation for l={l} not implemented.")

    def build_transform_matrix(self) -> np.ndarray:
        """
        Build the overall block‑diagonal transformation matrix X.

        The Cartesian basis functions are grouped by angular momentum on each atom.
        For each group (i.e., all functions with the same l on the same center),
        the appropriate transformation matrix is placed as a block on the diagonal.
        The order of blocks follows the original order of basis functions.

        Returns
        -------
        np.ndarray
            Transformation matrix X of shape (n_sph, n_cart), where n_sph is the total
            number of spherical harmonic basis functions and n_cart is the number of
            Cartesian basis functions. Multiplying a Cartesian integral matrix (n_cart, n_cart)
            as X @ matrix @ X.T yields the corresponding matrix in the spherical basis.
        """
        blocks = []
        func_idx = 0
        while func_idx < len(self.basis_functions):
            x, y, z = self.basis_functions[func_idx].angular
            l = x + y + z
            blocks.append(self._get_matrix(l))
            # Number of Cartesian functions for this l is (l+1)*(l+2)//2.
            # Advance the index to the start of the next group.
            func_idx += (l + 2) * (l + 1) // 2
        # Create a block diagonal matrix from all blocks.
        transform_matrix = scipy.linalg.block_diag(*blocks)
        return transform_matrix

if __name__ == "__main__":
    # Test the transformation matrix construction using a water molecule
    # with a basis set that includes d‑functions (6-31G(d,p)) to validate d‑block handling.

    # Define the water molecule with coordinates in angstroms.
    atom1 = Atom('O', np.array([-0.05591054, 2.17252383, 0.00000000]))
    atom2 = Atom('H', np.array([0.90408946, 2.17252383, 0.00000000]))
    atom3 = Atom('H', np.array([-0.37636513, 3.07745966, 0.00000000]))

    h2o_molecule = Molecule([atom1, atom2, atom3], charge=0, multiplicity=1)

    basis_set_manager = BasisSetManager("6-31g(d,p)")
    basis_functions = []
    for atom in h2o_molecule.atoms:
        shells = basis_set_manager.get_basis_for_atom(atom.symbol)
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

    # Build the transformation matrix.
    cart2sph = CartToSph(basis_functions)
    X = cart2sph.build_transform_matrix()

    print(len(basis_functions)) # Expected: number of Cartesian basis functions.
    print(X.shape) # Expected: (n_sph, n_cart) with n_sph < n_cart if d‑functions are present.

