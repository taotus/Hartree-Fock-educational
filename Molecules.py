import numpy as np
from typing import List, Dict, Tuple, Optional


PeriodicTable = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6,
    'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14,
    'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27,
}


# ==================== Molecular Structure ====================
class Atom:
    """原子类"""

    def __init__(self, symbol: str, position: np.ndarray, charge: Optional[float] = None):
        self.symbol = symbol
        self.position = np.asarray(position, dtype=float)
        self.position /= 0.5291772109

        # 根据元素符号确定核电荷数
        if charge is None:
            self.charge = self._get_nuclear_charge(symbol)
        else:
            self.charge = charge

    @staticmethod
    def _get_nuclear_charge(symbol: str) -> float:
        """根据元素符号获取核电荷数"""
        periodic_table = PeriodicTable
        return periodic_table.get(symbol, 0)


class Molecule:
    """分子类"""

    def __init__(self, atoms: List[Atom], charge: int = 0, multiplicity: int = 1):
        self.atoms = atoms
        self.charge = charge
        self.multiplicity = multiplicity
        self.natoms = len(atoms)

        # 分子中的电子数目
        self.nelectrons = self.count_electrons()

    def count_electrons(self) -> int:
        ne = 0
        for atom in self.atoms:
            ne += atom.charge
        ne -= self.charge
        return ne

    @property
    def nuclei_array(self) -> np.ndarray:
        return np.array([atom.position for atom in self.atoms])

    @property
    def nuclei(self) -> List[Dict]:
        """获取核信息列表"""
        return [
            {"position": atom.position, "charge": atom.charge}
            for atom in self.atoms
        ]

    @property
    def nuclear_repulsion_energy(self) -> float:
        """计算核排斥能"""
        energy = 0.0
        for i in range(self.natoms):
            for j in range(i + 1, self.natoms):
                R_ij = np.linalg.norm(self.atoms[i].position - self.atoms[j].position)
                energy += self.atoms[i].charge * self.atoms[j].charge / R_ij
        return energy