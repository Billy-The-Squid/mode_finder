# Mode Finder Package

# k_basis


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
   * Reorganize arguments for make_h
   * Make `L` not global
   * Rename `main` and separate it into multiple functions
 * Sensible 3D plotting for `k_basis.main`.