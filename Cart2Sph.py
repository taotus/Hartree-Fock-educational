import numpy as np
import scipy
from typing import List, Tuple

from Gaussian import BasisFunction
from Molecules import Atom, Molecule


class CartToSph:
    s_matrix = np.array(
        [[1]]
    )
    p_matrix = np.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]
    )
    d_matrix = np.array(
        [[0, 0, 0, 0.8660254038, -0.8660254038, 0],
         [0, 1.732050808, 0, 0, 0, 0],
         [0, 0, 0, -0.5, -0.5, 1],
         [0, 0, 1.732050808, 0, 0, 0],
         [1.732050808, 0, 0, 0, 0, 0]]
    )

    def __init__(self, basis_functions: List[BasisFunction]):
        self.basis_functions = basis_functions
        self.lmax = self._get_max_l()

    def _clip_basis_functions(self):

        for func in self.basis_functions:
            func_l = []
            x, y, z = func.angular
            l = x + y + z
            if len(func_l) == 0:
                func_l.append(func)

    def _get_matrix(self, l) -> np.ndarray:
        if l == 0:
            return self.s_matrix
        elif l == 1:
            return self.p_matrix
        elif l == 2:
            return self.d_matrix

    def _get_max_l(self) -> int:
        """获取基函数中最大的角动量量子数"""
        max_l = 0
        for func in self.basis_functions:
            x, y, z = func.angular
            l = x + y + z
            if l > max_l:
                max_l = l
        return max_l

    def build_transform_matrix(self) -> np.ndarray:
        blocks = []
        func_idx = 0
        while func_idx < len(self.basis_functions):
            x, y, z = self.basis_functions[func_idx].angular
            l = x + y + z
            blocks.append(self._get_matrix(l))
            func_idx += (l + 2) * (l + 1) // 2  # 跳过当前l的所有函数

        transform_matrix = scipy.linalg.block_diag(*blocks)
        return transform_matrix

if __name__ == "__main__":
    atom1 = Atom('O', np.array([-0.05591054, 2.17252383, 0.00000000]))
    atom2 = Atom('H', np.array([0.90408946, 2.17252383, 0.00000000]))
    atom3 = Atom('H', np.array([-0.37636513, 3.07745966, 0.00000000]))
    # atom4 = Atom('Fe', np.array([0.00000000, 0.00000000, 0.00000000]))

    h2o_molecule = Molecule([atom1, atom2, atom3], charge=0, multiplicity=1)

