import numpy as np
from grid.onedgrid import GaussChebyshev
from grid.rtransform import BeckeRTransform
from grid.molgrid import MolGrid
from grid.becke import BeckeWeights
from grid.atomgrid import AtomGrid


from Molecules import Atom, Molecule


class MoleculeGrid:
    def __init__(self, molecule: Molecule, n_radial: int = 10, preset: str = "medium"):
        self.molecule = molecule
        self.atoms = molecule.atoms
        self.n_radial = n_radial
        self.nuclei = molecule.nuclei
        self.preset = preset

    def build_molecule_grid(self) -> MolGrid:

        oned_grid = GaussChebyshev(npoints=self.n_radial)

        radial_grid = BeckeRTransform(0.0, R=1.5).transform_1d_grid(oned_grid)

        atom_numbers = np.array([atom.charge for atom in self.atoms])
        atom_coords = np.array([atom.position for atom in self.atoms])

        # 使用预设精度 "fine" 生成分子网格
        mol_grid = MolGrid.from_preset(
            atnums=atom_numbers,
            atcoords=atom_coords,
            rgrid=radial_grid,  # 共享的径向网格
            preset=self.preset,  # 控制角向精度，可选 "coarse", "medium", "fine", "veryfine", "ultrafine"
            aim_weights=BeckeWeights(),  # 使用 Becke 权重进行原子划分
            store=True  # 存储单个原子网格信息，便于后续分析
        )
        return mol_grid

if __name__ == "__main__":
    atom1 = Atom('O', np.array([-0.05591054, 2.17252383, 0.00000000]))
    atom2 = Atom('H', np.array([0.90408946, 2.17252383, 0.00000000]))
    atom3 = Atom('H', np.array([-0.37636513, 3.07745966, 0.00000000]))

    h2o_molecule = Molecule([atom1, atom2, atom3], charge=0, multiplicity=1)
    mol_grid = MoleculeGrid(h2o_molecule).build_molecule_grid()
    print(mol_grid.points.shape)
    print(mol_grid.weights.shape)

