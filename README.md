# A From‑Scratch Hartree–Fock Implementation

**PyHF** is a pure‑Python implementation of the Hartree–Fock (HF) method for molecular electronic structure calculations. It is written from scratch with **readability and education** as the primary goals – no black‑box libraries, just NumPy and SciPy to see exactly how HF works under the hood.

## Features

-  **Gaussian basis sets** – Cartesian orbitals of any angular momentum.
-  **Obara–Saika recursion** – Evaluation of two‑electron repulsion integrals via OS recursion.
-  **One‑electron integrals** – Overlap, kinetic energy, nuclear attraction.
-  **Two‑electron integrals** – Precomputed and reused in the SCF cycle.
-  **SCF iteration** – Use core Hamiltonian as initial guess and solve Roothaan equation
-  **Modular design** – Build molecules in the way like Gaussian gjf files.
-  **Object-Oriented** - Every function is encapsulated in its own class

## Dependencies

- Python 3.8+
- NumPy
- SciPy

