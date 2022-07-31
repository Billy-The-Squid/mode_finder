# Mode Finder Package
This package was designed to simulate a particular minimal 
topological superconductor model. This is not being actively
maintained. This package can be found at 
https://github.com/Billy-The-Squid/mode_finder

# Quick Start
From python, import the package and, specifically, the 
toy_model file
```python
import mode_finder as mf
from toy_model import *
```
Create a 50x50 lattice with a set of parameters
```python
latt = setup_system(50, 0.5, -1.8, -0.7, 0.1, 0.1)
```
Introduce a dislocation at the point (20, 20), extending in 
the -y direction
```python
latt.dislocation(20, 20, 3)
```
Then calculate the 10 closest-to-zero eigenvalues
```python
latt.calculate_eigenvalues_sparse(10)
```
The values have now been stored in `latt`, and can be 
numerically accessed through `latt.eigenvalues`. You can 
plot the spatial probability distribution of one eigenstate 
with 
```python
latt.plot_eigenvalue(0)
```


# Files
Each of the files in the small package has its own intended
purpose. See the documentation for individual functions for
more detail. 

## lattice
The most organized of the files, this sets up the Lattice
class for general square lattices with only on-site and 
nearest-neighbor terms. Initialize the Lattice object with
the size and degrees of freedom, then add diagonal and 
hopping terms. Once those are set up, you can introduce a 
dislocation. Finally, you can calculate some of the 
eigenvalues of the system and plot them in various ways. 

## toy_model
A wrapper for `lattice` that implements the minimal TSC for
a given set of parameters.

## k_basis
Disconnected from `lattice` and `toy_model`, contains 
functions for plotting band structures for the minimal TSC
model and doing various phase calculations. 

## phase_test
A mostly last-minute file with a handful of functions for 
scanning through sets of parameters. 


# Further improvements
In no particular order:
 * Make the lattice class more robust to allow for more/fewer
    degrees of freedom, more options for DoFs, etc. 
 * Better support for multiple dislocations in a lattice
   * Same site_cost for each removed site
 * Implement bdg_h as a scipy sparse matrix
 * Make options for ribbon_dir more sensible
 * Rename `calculate_eigenvalues` and 
   `calculate_eigenvalues_sparse` to suggest which one is 
   more practical
 * Option for finite-length dislocations
 * Make dislocations play nice with periodic BCs
 * Clear up ambiguity about `dofs` vs `dofs_dict`
 * Make everything in `k_basis` behave much more intuitively.
   * Reorganize arguments for `make_h` and `kappa` functions
   * Make `L` not global
   * Rename `main` and separate it into multiple functions
 * Sensible 3D plotting for `k_basis.main`.
 * Signed/unsigned variants of kappa functions
 * Step sizes for scan functions
 * Rename/reorganize various scan functions
 * `scan_parameters` doesn't need `v` or `delta`
 * Make `phase_test` functions take sensible arguments