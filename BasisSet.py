import json
import basis_set_exchange as bse
from typing import List, Dict

from Molecules import PeriodicTable


# ==================== Basis Set Manager ====================
class BasisSetManager:
    """
    Manager for basis set data using the Basis Set Exchange (BSE) library.
    Loads basis set information in JSON format and provides methods to retrieve
    basis functions (exponents, coefficients, angular momentum) for a given atom.
    """
    def __init__(self, basis_name: str = 'STO-3G'):
        """
        Initialise the basis set manager.

        Parameters
        ----------
        basis_name : str
            Name of the basis set (e.g., 'STO-3G', '3-21g').
        """
        self.basis_name = basis_name
        self._load_basis_set()

    def _load_basis_set(self):
        """Load the basis set data from the Basis Set Exchange as a JSON dictionary."""
        # Get basis set in JSON format
        basis_str = bse.get_basis(self.basis_name, fmt='json')
        self.basis_data = json.loads(basis_str)

    def get_basis_for_atom(self, atom_symbol: str) -> List[Dict]:
        """
        Return the basis set information for a specific atom.

        Parameters
        ----------
        atom_symbol : str
            Chemical symbol of the atom (e.g., 'H', 'O').

        Returns
        -------
        List[Dict]
            A list of shell dictionaries. Each shell contains:
            - "angular": list of (lx, ly, lz) tuples representing Cartesian components.
            - "exponents": list of primitive exponents.
            - "coefficients": list of contraction coefficients corresponding to the exponents.
        """
        # Convert element symbol to atomic number
        atomic_number = self._symbol_to_z(atom_symbol)

        if str(atomic_number) not in self.basis_data["elements"]:
            raise ValueError(f"Element {atom_symbol} not found in basis set {self.basis_name}")

        element_data = self.basis_data["elements"][str(atomic_number)]
        shells = []

        # Iterate over each electron shell defined for this element
        for shell in element_data["electron_shells"]:
            # Angular momentum of the shell (a list if the shell contains multiple components)
            angular_momentum = shell["angular_momentum"]

            # For each angular momentum component in the shell (e.g., separate p orbitals)
            for i, l in enumerate(angular_momentum):
                exponents = shell["exponents"] # list of primitive exponents
                coeffs = shell["coefficients"][i] # contraction coefficients for this component

                # Convert the angular momentum quantum number l into Cartesian (lx, ly, lz) tuples
                shell_info = {
                    "angular": self._l_to_xyz(l),
                    "exponents": [float(e) for e in exponents],
                    "coefficients": [float(c) for c in coeffs]
                }
                shells.append(shell_info)
        return shells

    @staticmethod
    def _symbol_to_z(symbol: str) -> int:
        """
        Convert an element symbol to its atomic number.

        Parameters
        ----------
        symbol : str
            Chemical symbol (e.g., 'H', 'O').

        Returns
        -------
        int
            Atomic number.
        """
        # Use predefined periodic table mapping
        periodic_table = PeriodicTable
        return periodic_table.get(symbol, 0)

    @staticmethod
    def _l_to_xyz(l: int) -> list:
        """
        Convert an angular momentum quantum number l into a list of Cartesian exponent tuples.

        For a given l, this returns all possible combinations (lx, ly, lz) such that lx+ly+lz = l.
        Currently implemented for s, p, and d orbitals.

        Parameters
        ----------
        l : int
            Angular momentum quantum number (0 for s, 1 for p, 2 for d).

        Returns
        -------
        list of tuple
            List of (lx, ly, lz) tuples representing Cartesian Gaussian angular parts.

        Raises
        ------
        NotImplementedError
            If l > 2 (f‑orbitals and higher are not implemented).
        """
        if l == 0:  # s-type
            return [(0, 0, 0)]
        elif l == 1:  # p-type
            return [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        elif l == 2:
            return [(1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 0, 0), (0, 2, 0), (0, 0, 2)]
        else:
            raise NotImplementedError(f"Angular momentum l={l} not implemented (only s, p, d supported).")
