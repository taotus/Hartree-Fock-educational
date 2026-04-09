import numpy as np
from grid.onedgrid import GaussChebyshev
from grid.rtransform import BeckeRTransform
from grid.molgrid import MolGrid
from grid.becke import BeckeWeights

from Molecules import Atom, Molecule


class MoleculeGrid:
    """Molecular grid generator for numerical integration."""

    def __init__(self, molecule: Molecule, n_radial: int = 10, preset: str = "medium"):
        """
        Parameters:
        -----------
        molecule : Molecule
            The molecule object containing atoms and their properties.
        n_radial : int
            Number of radial grid points per atom.
        preset : str
            Angular grid precision preset (e.g., "coarse", "medium", "fine").
        """
        self.molecule = molecule
        self.atoms = molecule.atoms
        self.n_radial = n_radial
        self.nuclei = molecule.nuclei
        self.preset = preset

    def build_molecule_grid(self) -> MolGrid:
        """Construct and return a molecular grid using Becke partitioning."""
        # 1D radial grid: Gauss-Chebyshev quadrature
        oned_grid = GaussChebyshev(npoints=self.n_radial)

        # Transform to radial coordinate with Becke's transformation
        radial_grid = BeckeRTransform(0.0, R=1.5).transform_1d_grid(oned_grid)

        # Extract atomic numbers and coordinates for all atoms
        atom_numbers = np.array([atom.charge for atom in self.atoms])
        atom_coords = np.array([atom.position for atom in self.atoms])

        # Generate molecular grid using preset angular precision
        mol_grid = MolGrid.from_preset(
            atnums=atom_numbers,
            atcoords=atom_coords,
            rgrid=radial_grid,          # Shared radial grid
            preset=self.preset,         # Angular precision: "coarse", "medium", "fine", "veryfine", "ultrafine"
            aim_weights=BeckeWeights(), # Becke partitioning weights
            store=True                  # Store atomic subgrids for analysis
        )
        return mol_grid


if __name__ == "__main__":
    # Define water molecule (H2O) coordinates
    atom1 = Atom('O', np.array([-0.05591054, 2.17252383, 0.00000000]))
    atom2 = Atom('H', np.array([0.90408946, 2.17252383, 0.00000000]))
    atom3 = Atom('H', np.array([-0.37636513, 3.07745966, 0.00000000]))

    h2o_molecule = Molecule([atom1, atom2, atom3], charge=0, multiplicity=1)

    # Build molecular grid and print its properties
    mol_grid = MoleculeGrid(h2o_molecule).build_molecule_grid()
    print(mol_grid.points.shape)   # Shape of grid points (n_points, 3)
    print(mol_grid.weights.shape)  # Shape of integration weights (n_points,)