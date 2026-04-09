import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
from itertools import combinations

from Molecules import Atom, Molecule
from HF import HartreeFock

np.set_printoptions(
    threshold=500,
    linewidth=500,
    precision=4
)


class CI(HartreeFock):
    def __init__(self, molecule: Molecule, truncated_states: int=None, basis_name: str = 'STO-3G'):
        super().__init__(molecule, basis_name)
        self.truncated_states = truncated_states
        self.scf_iteration()
        self.H_mo = self.C.T @ self.H @ self.C
        self.ERI_mo = np.einsum('pqrs,pi,qj,rk,sl->ijkl', self.SphERI, self.C, self.C, self.C, self.C)
        print("initialization of FullCI done.")
        self.num_determinants = 0
        self.correlation_energy = 0.
        self.excited_states = None
        self.H_matrix = self._build_determinant_matrix()
        self.configuration_coefficients = None

    @staticmethod
    def _check_differ_of_excitation(orb_array1: np.ndarray, orb_array2: np.ndarray) -> Tuple[List[Tuple[int, int]], int]:
        arr1 = orb_array1.copy()
        arr2 = orb_array2.copy()

        diff = []
        permutation = 0
        for i in range(len(arr1)):
            if arr1[i] == arr2[i]:
                continue
            else:
                for j in range(i, len(arr2)):
                    if arr1[i] == arr2[j]:
                        arr2[i], arr2[j] = arr2[j], arr2[i]
                        permutation += 1
                        break
                if arr1[i] != arr2[i]:
                    for k in range(i, len(arr1)):
                        if arr1[k] == arr2[i]:
                            arr1[k], arr1[i] = arr1[i], arr1[k]
                            permutation += 1
                            break
                    if arr1[i] != arr2[i]:
                        diff.append((arr1[i], arr2[i]))
        return diff, permutation

    @staticmethod
    def _delta_spin(spin_orb1: int, spin_orb2: int) -> int:
        if spin_orb1 % 2 == spin_orb2 % 2:
            return 1
        else:
            return 0

    # 索引2i+1表示自旋向下，索引2i表示自旋向上
    def _compute_determinant_matrix_element(self, spin_orbs1: np.ndarray, spin_orbs2: np.ndarray) -> float:

        differ_list, permutation = self._check_differ_of_excitation(spin_orbs1, spin_orbs2)
        #print(differ_list)
        n_differ = len(differ_list)

        if n_differ == 0:
            core_h = 0.
            v_ee = 0.
            for i_so in spin_orbs1:
                i = i_so // 2
                core_h += self.H_mo[i, i]
                for j_so in spin_orbs2:
                    j = j_so // 2
                    v_ee += self.ERI_mo[i, i, j, j] - self.ERI_mo[i, j, i, j] * self._delta_spin(i_so, j_so)
            v_ee *= 0.5
        elif n_differ == 1:
            m_so, p_so = differ_list[0]
            m = m_so // 2
            p = p_so // 2
            core_h = self.H_mo[m, p] * self._delta_spin(m_so, p_so)
            v_ee = 0.
            for i_so in spin_orbs1:
                i = i_so // 2
                v_ee += self.ERI_mo[m, p, i, i] * self._delta_spin(m_so, p_so) - (
                        self.ERI_mo[m, i, i, p] * self._delta_spin(m_so, i_so) * self._delta_spin(p_so, i_so)
                )
        elif n_differ == 2:
            core_h = 0.
            m_so, p_so = differ_list[0]
            n_so, q_so = differ_list[1]
            m, p, n, q = m_so // 2, p_so // 2, n_so // 2, q_so // 2
            v_ee = self.ERI_mo[m, p, n, q] * self._delta_spin(m_so, p_so) * self._delta_spin(n_so, q_so) - (
                   self.ERI_mo[m, q, n, p] * self._delta_spin(m_so, q_so) * self._delta_spin(n_so, p_so)
            )
        else:
            core_h = 0.
            v_ee = 0.

        return (core_h + v_ee) * np.power(-1, permutation)

    def _build_excitation_states(self) -> Dict[int, List[List[Tuple[int, int]]]]:

        excit_list = []
        for i in range(self.nocc * 2):
            for a in range(self.nocc * 2, self.n_sph * 2):
                excit_list.append((i, a))

        excitated_states = {}
        if self.truncated_states is not None:
            multiplicative = min(self.molecule.nelectrons, self.truncated_states)
        else:
            multiplicative = self.molecule.nelectrons
        for m in range(1, multiplicative + 1):
            valid_excits = []
            for comb in combinations(excit_list, m):
                # 检查占据轨道互异且虚轨道互异
                occ_set = set()
                virt_set = set()
                valid = True
                for i, a in comb:
                    if i in occ_set or a in virt_set:
                        valid = False
                        break
                    occ_set.add(i)
                    virt_set.add(a)
                if valid:
                    valid_excits.append(list(comb))
            excitated_states[m] = valid_excits

        return excitated_states

    def _build_determinant_matrix(self):
        self.excits_states = self._build_excitation_states()
        excits_list = [[(0, 0)]]
        excit_ways = 0
        for m, excit in self.excits_states.items():
            excit_ways += len(excit)
            excits_list.extend(excit)
        print(f"激发方式数: {excit_ways}")

        determinants = []
        for excit in excits_list:
            spin_orbs = np.arange(self.nocc * 2)
            for excit_pair in excit:
                spin_orbs[excit_pair[0]] = excit_pair[1]
            spin_orbs = set(spin_orbs)
            if spin_orbs not in determinants:
                determinants.append(spin_orbs)

        self.num_determinants = len(determinants)
        print(f"激发态数目: {self.num_determinants}")

        H_matrix = np.zeros([self.num_determinants, self.num_determinants])
        for i in range(self.num_determinants):
            for j in range(i, self.num_determinants):
                H_matrix[i, j] = H_matrix[j, i] = self._compute_determinant_matrix_element(
                    np.array(list(determinants[i])), np.array(list(determinants[j])))

        return H_matrix

    def compute_correlation_energy(self):
        ehf_matrix = np.identity(self.num_determinants) * self.electron_energy
        ci_matrix = self.H_matrix - ehf_matrix
        epsilon, coefficients = np.linalg.eigh(ci_matrix)
        self.configuration_coefficients = coefficients
        self.correlation_energy = epsilon[0]
        return self.correlation_energy


if __name__ == "__main__":
    atom1 = Atom('O', np.array([-0.05591054, 2.17252383, 0.00000000]))
    atom2 = Atom('H', np.array([0.90408946, 2.17252383, 0.00000000]))
    atom3 = Atom('H', np.array([-0.37636513, 3.07745966, 0.00000000]))

    h2o_molecule = Molecule([atom1, atom2, atom3], charge=0, multiplicity=1)

    cisd = CI(h2o_molecule, basis_name='STO-3G')

    print(cisd.H_matrix)
    print(cisd.compute_correlation_energy())

