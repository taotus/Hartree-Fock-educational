import json
import basis_set_exchange as bse
from typing import List, Dict

from Molecules import PeriodicTable


# ==================== Basis Set Manager ====================
class BasisSetManager:
    """基组管理器"""
    def __init__(self, basis_name: str = 'STO-3G'):
        self.basis_name = basis_name
        self._load_basis_set()

    def _load_basis_set(self):
        """从BSE加载基组"""
        # 获取所有元素的基组信息
        basis_str = bse.get_basis(self.basis_name, fmt='json')
        self.basis_data = json.loads(basis_str)

    def get_basis_for_atom(self, atom_symbol: str) -> List[Dict]:
        """获取特定原子的基组信息"""
        atomic_number = self._symbol_to_z(atom_symbol)

        if str(atomic_number) not in self.basis_data["elements"]:
            raise ValueError(f"基组 {self.basis_name} 中没有元素 {atom_symbol} 的信息")

        element_data = self.basis_data["elements"][str(atomic_number)]
        shells = []

        for shell in element_data["electron_shells"]:
            # 获取角动量信息
            angular_momentum = shell["angular_momentum"]

            # 对于每个角动量分量
            for i, l in enumerate(angular_momentum):
                exponents = shell["exponents"]
                coeffs = shell["coefficients"][i]

                # 转换为标准格式
                shell_info = {
                    "angular": self._l_to_xyz(l),
                    "exponents": [float(e) for e in exponents],
                    "coefficients": [float(c) for c in coeffs]
                }
                shells.append(shell_info)
        return shells

    @staticmethod
    def _symbol_to_z(symbol: str) -> int:
        """元素符号到原子序数"""
        periodic_table = PeriodicTable
        return periodic_table.get(symbol, 0)

    @staticmethod
    def _l_to_xyz(l: int) -> list:
        """角动量量子数l转换为(x, y, z)方向"""
        # 简化处理：仅处理s和p轨道
        if l == 0:  # s轨道
            return [(0, 0, 0)]
        elif l == 1:  # p轨道
            return [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        elif l == 2:
            return [(1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 0, 0), (0, 2, 0), (0, 0, 2)]
        else:
            raise NotImplementedError(f"角动量量子数 l={l} 尚未实现")
