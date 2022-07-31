import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
import scipy.sparse as spr
from time import perf_counter

paulis = [
    np.array([[1, 0], [0, 1]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, -1j], [1j, 0]]),
    np.array([[1, 0], [0, -1]])
]


def kron(arg_list):
    """
    A Kronecker product of an arbitrary number of matrices.

    :param arg_list: A list of numpy arrays.
    :return: A numpy array.
    """
    if len(arg_list) < 1:
        raise Exception("At least one argument must be provided.")
    if len(arg_list) == 1:
        return arg_list[0]
    else:
        ret_val = 1
        for mat in arg_list:
            ret_val = np.kron(ret_val, mat)
    return ret_val


class Lattice:
    """
    A class that sets up a square crystal lattice with superconducting terms.

    L: The length of one edge of the lattice
    dofs: A list of strings. A list of the degrees of freedom (e.g., spin, orbital) for all sites.
    dofs_dict: A dictionary mapping strings to lists of strings. Maps the names of degrees of freedom to the possible
        values of that DoF.
    bdg_h: A numpy array. The BdG Hamiltonian for the system. Each row/column corresponds to a site on the lattice, an
        electron or hole, and a possible combination of parameters.
    eigenvalues: A numpy array of floats. The set of eigenvalues of the system, sorted from least magnitude to
        greatest.
    eigenvectors: A 2D numpy array of floats. Each column corresponds to an eigenvector of bdg_h, and columns are
        sorted in the same order as eigenvalues.
    removed_site_count: If a dislocation has been introduced into the system, tracks the number of sites that have been
        removed from the lattice.
    ribbon_dir: False, "x", "y", or "xy". Indicates the direction(s) in which the system has periodic boundary
        conditions.
    """
    def __init__(self, L: int, dofs: dict, ribbon_dir=False):
        """
        Creates an empty matrix representing the BdG Hamiltonian (in r-basis) for the system.
        We assume the system is a square 2D lattice.

        :param L: The length of one side of the lattice, or the number of sites along one axis.
        :param dofs: The degrees of freedom at each site. This should be a dictionary with dof names
            (as strings) mapping to lists of possible values. The dictionary should not be empty and
            each list should have more than one entry.
        :param ribbon_dir: False to have open boundaries, "x" or "y" for a x-extended/y-extended ribbon geometry,
            "xy" for closed boundaries.
        """
        # Some important variables
        self.L = L
        self.dofs_dict = dofs
        self.dofs = list()
        # Options per site
        self._options_per_site = 1
        for i in dofs:
            self._options_per_site *= len(dofs[i])
            self.dofs.append(i)
        count = self._options_per_site * self.L * self.L * 2  # The 2 accounts for antiparticles
        # Initialize the Hamiltonian
        self.bdg_h = numpy.zeros((count, count), dtype=np.complex64)
        # Eigenvalues have not yet been calculated
        self.eigenvectors = None
        self.eigenvalues = None
        self._eigen_up_to_date = False
        self.removed_site_count = 0
        # Stores a sparse form of the Hamiltonian to reduce redundant computation
        self._sparse_h = None
        # A list of site costs in a dislocated matrix.
        self._site_costs = []
        self.ribbon_dir = ribbon_dir

    def get_index(self, x: int, y: int, dofs: dict, particle=True):
        """
        Returns the integer index of the specified site with the specified degree of freedom. (x,y) = (0,0) is in a
            corner. The returned index can be used to access a row or column of bdg_h.
        :param x: The x coordinate of the site.
        :param y: The y coordinate of the site.
        :param dofs: A dictionary mapping the strings labeling the dofs to the index of their value in the dof_dict
            list.
        :param particle: True for the particle part of the matrix, false for the hole part.
        :return: The integer index of the site specified.
        """
        if self.ribbon_dir in [False, "y"] and (x < 0 or x >= self.L):
            raise IndexError("x index exceeds size of array")
        if self.ribbon_dir in [False, "x"] and (y < 0 or y >= self.L):
            raise IndexError("y index exceeds size of array")
        x = x % self.L
        y = y % self.L
        index = 0
        index += y
        index = index * self.L + x
        for i in self.dofs:
            index = index * len(self.dofs_dict[i]) + dofs[i]
        if not particle:
            index += self._options_per_site * (self.L ** 2)
        return index

    def get_parameters(self, index: int):
        """
        Returns the coordinates, particle/antiparticle, and values of the dofs.
        :param index: The index of the specified state.
        :return: A tuple. Index 0 is a tuple with the x and y coordinates. Index 1 gives a boolean indicating whether
            the state is a particle. Index 2 is a dictionary with the indices of each of the dofs.
        """
        particle = index < self._options_per_site * (self.L ** 2)
        y = (index // (self._options_per_site * self.L)) % self.L
        x = (index // self._options_per_site) % self.L
        dofs = {}
        block_size = self._options_per_site
        for i in self.dofs:
            options = len(self.dofs_dict[i])
            block_size = block_size / options
            dofs[i] = (index // block_size) % options
        return (x, y), particle, dofs

    def add_diagonal(self, term, pauli_index, pauli_indices, verbose=False):
        """
        Adds diagonal terms (x = y) to bdg_h.

        :param term: The term to be added at each component of the matrix. Will be multiplied by the relevant Pauli
            matrices.
        :param pauli_index: Integer (0, 1, 2, 3). The index of the Pauli matrix that determines the contributions in
            each block of the BdG Hamiltonian. E.g., 2 will only insert `term` in the upper right (multiplied by -1j)
            and lower left (multiplied by 1j) blocks of bdg_h.
        :param pauli_indices: Dictionary mapping strings (from the dof list) to integers between 0 and 3. Associates
            each DoF to a Pauli matrix that determines whether (and how) different values of the DoF interact.
        :param verbose: Boolean. True prints the amount of time required to add the terms.
        """
        self._eigen_up_to_date = False
        self._sparse_h = None
        start_t = perf_counter()
        # Calculate what each block will look like
        on_site_term = term * kron([paulis[pauli_indices[dof]] for dof in self.dofs])
        # Some important values
        init = {dof: 0 for dof in self.dofs}
        ph_p_mat = paulis[pauli_index]
        blocks = [(0, 0), (1, 0), (0, 1), (1, 1)]

        # Iterate over the sites
        for x in range(self.L):
            for y in range(self.L):
                site_i = [
                    self.get_index(x, y, init, particle=True),
                    self.get_index(x, y, init, particle=False)
                ]
                for dest, source in blocks:
                    # Add the on-site terms
                    self.bdg_h[site_i[dest]:(site_i[dest] + self._options_per_site),
                    site_i[source]:(site_i[source] + self._options_per_site)] += \
                        on_site_term.copy() * ph_p_mat[dest, source]
        if verbose:
            print("Time to add hopping term: " + str(perf_counter() - start_t))

    def add_hopping(self, term, pauli_index, pauli_indices, direction, verbose=False):
        """
        Adds a c_x c_{x+-1} term to the matrix. Preserves Hermiticity but will not add the opposite-direction hopping
            term if pauli_index is 1 or 2.
        :param term: The term to be added at each component of the matrix. Will be multiplied by the relevant Pauli
            matrices.
        :param pauli_index: Integer (0, 1, 2, 3). The index of the Pauli matrix that determines the contributions in
            each block of the BdG Hamiltonian. E.g., 2 will only insert `term` in the upper right (multiplied by -1j)
            and lower left (multiplied by 1j) blocks of bdg_h.
        :param pauli_indices: Dictionary mapping strings (from the dof list) to integers between 0 and 3. Associates
            each DoF to a Pauli matrix that determines whether (and how) different values of the DoF interact.
        :param direction: 0 for x->x+1, 1 for y->y+1, 2 for x->x-1, 3 for y->y-1
        :param verbose: Boolean. True prints the amount of time required to add the terms.
        """
        self._eigen_up_to_date = False
        self._sparse_h = None

        start_t = perf_counter()
        # The blocks to insert at each site.
        on_site_term = term * kron([paulis[pauli_indices[dof]] for dof in self.dofs])
        if pauli_index == 2:
            on_site_term = -1j * on_site_term
        adjoint = on_site_term.conj().transpose()
        # Some important values
        init = {dof: 0 for dof in self.dofs}
        ph_p_mat = paulis[pauli_index]
        dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        # Iterate over the sites
        for x in range(self.L):
            for y in range(self.L):
                # Indices for all blocks
                site_i = [
                    self.get_index(x, y, init, particle=True),
                    self.get_index(x, y, init, particle=False)
                ]
                # Check for edges/wrapping
                try:
                    neighbor_i = [
                        self.get_index(x + dirs[direction][0], y + dirs[direction][1], init, particle=True),
                        self.get_index(x + dirs[direction][0], y + dirs[direction][1], init, particle=False)
                    ]
                except IndexError:
                    continue

                if pauli_index in [0, 3]:
                    # Iterate through the two relevant blocks
                    for dest, source in [(0, 0), (1, 1)]:
                        # Add the site -> neighbor terms
                        self.bdg_h[neighbor_i[dest]:(neighbor_i[dest] + self._options_per_site),
                        site_i[source]:(site_i[source] + self._options_per_site)] += \
                            on_site_term.copy() * ph_p_mat[dest, source]
                        # Add the neighbor -> site terms
                        self.bdg_h[site_i[dest]:(site_i[dest] + self._options_per_site),
                        neighbor_i[source]:(neighbor_i[source] + self._options_per_site)] += \
                            adjoint.copy() * ph_p_mat[dest, source]
                elif pauli_index in [1, 2]:
                    # Add the site -> neighbor term
                    self.bdg_h[neighbor_i[0]:(neighbor_i[0] + self._options_per_site),
                    site_i[1]:(site_i[1] + self._options_per_site)] += \
                        on_site_term.copy()
                    # Add the neighbor -> site terms
                    self.bdg_h[site_i[1]:(site_i[1] + self._options_per_site),
                    neighbor_i[0]:(neighbor_i[0] + self._options_per_site)] += \
                        adjoint.copy()
        if verbose:
            print("Time to add diagonal: " + str(perf_counter() - start_t))

    def get_block(self, index):
        """
        Returns view of a block of the matrix indicated by the index. 0 gives the c*c terms, 1 the c*c* terms, 2 the cc
            terms, and 3 the cc* terms.
        :param index: An integer between 0 and 3, inclusive.
        :return: A view of the bdg_h array in the specified block. Width will be half the width of the full Hamiltonian.
        """
        c = index % 2
        r = index // 2
        w = self._options_per_site * self.L * self.L
        return self.bdg_h[r * w:(r + 1) * w, c * w:(c + 1) * w]

    def calculate_eigenvalues(self, verbose=False):
        """
        If the eigenvalues or eigenvectors are not up-to-date, recalculates them and sorts them from least magnitude to
        greatest, stored in lattice.eigenvalues and lattice.eigenvectors. Ridiculously inefficient for large lattice.L.

        :param verbose: Boolean. True prints the amount of time required to add the terms.
        """
        init = perf_counter()
        if not (self._eigen_up_to_date is True):
            vals, vecs = np.linalg.eigh(self.bdg_h)
            indices = np.argsort(vals ** 2)
            self.eigenvalues = vals[indices]
            self.eigenvectors = vecs[:, indices]
            self._eigen_up_to_date = True
        if verbose:
            print("Time to calculate all eigenvalues: " + str(perf_counter() - init))

    def calculate_eigenvalues_sparse(self, count, verbose=False):
        """
        If the eigenvalues or eigenvectors are not up-to-date, recalculates some of them and sorts them from least
            magnitude to greatest, stored in lattice.eigenvalues and lattice.eigenvectors.

        :param count: The number of eigenvalues to calculate. Finds closest-to-zero first.
        :param verbose: Boolean. True prints the amount of time required to add the terms.
        """
        if count >= self.bdg_h.shape[0] - 1:
            self.calculate_eigenvalues()
            return
        init = perf_counter()
        if self._eigen_up_to_date is False or self._eigen_up_to_date < count:
            if self._sparse_h is None:
                self._sparse_h = spr.csr_matrix(self.bdg_h)
            if verbose:
                print("Time to generate sparse matrix: " + str(perf_counter() - init))
                next = perf_counter()
            vals, vecs = eigsh(self._sparse_h, k=count, sigma=0, which="LM")
            if verbose:
                print("Time to find some eigenvalues: " + str(perf_counter() - next))
            indices = np.argsort(vals ** 2)
            self.eigenvalues = vals[indices]
            self.eigenvectors = vecs[:, indices]
            self._eigen_up_to_date = count
        if verbose:
            print("Total time to calculate some eigenvalues: " + str(perf_counter() - init))

    def plot_eigenvector(self, index, color=True):
        """
        Plots the probability density of the wavefunction indicated by the eigenvector at the specified index.

        :param index: The index of the eigenvector in lattice.eigenvectors.
        :param color: Boolean. True to render the plot in color, False for grayscale.
        """
        if self._eigen_up_to_date is False or self._eigen_up_to_date < index:
            print("Eigenvectors have not been computed to this index.")
            return
        xs = np.arange(self.L)
        ys = np.arange(self.L)
        xx, yy = np.meshgrid(xs, ys)
        probs = np.zeros(xx.shape, dtype=float)
        vec = self.eigenvectors[:, index]
        for i in range(len(vec)):
            coords = self.get_parameters(i)[0]
            probs[coords[1], coords[0]] += (vec[i] * vec[i].conjugate()).real

        # Plot!
        if color:
            cmap = "viridis"
        else:
            cmap = "binary"
        plt.imshow(probs, cmap=cmap)
        plt.title("Spatial probability distribution (%dx%d lattice), energy = %f"
                  %(self.L, self.L, self.eigenvalues[index]))
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.colorbar()
        plt.show()

    def dislocation(self, x_start, y_start, direc, site_cost=10):
        """
        Creates a dislocation starting at the specified point and continuing in the specified direction by removing the
            sites from that site and continuing in that direction. The sites on either side of the missing sites will
            have hopping terms to each other.
        This should only be used once in a lattice, and never before adding additional hopping or on-site terms.

        :param x_start: Int. The x coordinate of the starting point.
        :param y_start: Int. The y coordinate of the starting point.
        :param direc: 0 for +x, 1 for +y, 2 for -x, 3 for -y. The direction in which the dislocation will propagate
            from the starting point.
        :param site_cost: Real Float. The value assigned to on-site terms. Recommended to be larger in magnitude than
            any other eigenvalue could realistically be, so as to separate the true spectrum of the system from the
            vestigal eigenstates that reside only on removed sites.
        """
        # NOTE: This code will NOT generalize if next-nearest-neighbor terms are introduced.
        self._eigen_up_to_date = False
        self._sparse_h = None

        # Establish the direction to procede
        dirs = np.array([
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1]
        ])
        inc = dirs[direc]
        sides = [dirs[direc - 1],
                 dirs[(direc + 1) % 4]]

        # Get a list of the sites that are to be removed
        cursor = np.array([x_start, y_start])
        missing_sites = []
        while 0 <= cursor[0] < self.L and 0 <= cursor[1] < self.L:
            missing_sites.append(cursor.copy())
            cursor += inc
        self.removed_site_count = len(missing_sites)
        # TODO: Combine this for loop with the one below.

        # An important dictionary we'll need.
        init = self.dofs_dict.copy()
        for dof in self.dofs:
            init[dof] = 0

        # iterate through the sites that are to be removed.
        for site in missing_sites:
            neighbor0 = site + sides[0]  # Coords of neighbor 0
            neighbor1 = site + sides[1]  # Coords of neighbor 1
            neighbor2 = site - inc  # Coords of neighbor 1
            start_i = [
                self.get_index(site[0], site[1], init, particle=True),
                self.get_index(site[0], site[1], init, particle=False)
            ]
            side0_i = [
                self.get_index(neighbor0[0], neighbor0[1], init, particle=True),
                self.get_index(neighbor0[0], neighbor0[1], init, particle=False)
            ]
            side1_i = [
                self.get_index(neighbor1[0], neighbor1[1], init, particle=True),
                self.get_index(neighbor1[0], neighbor1[1], init, particle=False)
            ]
            back_i = [
                self.get_index(neighbor2[0], neighbor2[1], init, particle=True),
                self.get_index(neighbor2[0], neighbor2[1], init, particle=False)
            ]

            # Remove on-site terms
            for i in range(2):
                self.bdg_h[start_i[i]:(start_i[i] + self._options_per_site),
                start_i[i]:(start_i[i] + self._options_per_site)] = \
                    site_cost * np.identity(self._options_per_site, dtype=np.complex64)

            # Iterate through each of the four blocks in the BdG Hamiltonian
            for dest, source in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                # # Remove the on-site terms
                # self.bdg_h[start_i[dest]:(start_i[dest] + self._options_per_site),
                # start_i[source]:(start_i[source] + self._options_per_site)] = \
                #     site_cost * np.identity(self._options_per_site, dtype=np.complex64)

                # Remove hopping terms between removed sites
                self.bdg_h[start_i[dest]:(start_i[dest] + self._options_per_site),
                back_i[source]:(back_i[source] + self._options_per_site)] = \
                    np.zeros((self._options_per_site, self._options_per_site), dtype=complex)
                self.bdg_h[back_i[dest]:(back_i[dest] + self._options_per_site),
                start_i[source]:(start_i[source] + self._options_per_site)] = \
                    np.zeros((self._options_per_site, self._options_per_site), dtype=complex)
                # Note: This also removes the connection between the starting site and its still-extant neighbor,
                # as intended

                # Re-arrange hopping term from neighbor 0 to gap
                to_term_0 = self.bdg_h[start_i[dest]:(start_i[dest] + self._options_per_site),
                            side0_i[source]:(side0_i[source] + self._options_per_site)].copy()  # Copy
                self.bdg_h[start_i[dest]:(start_i[dest] + self._options_per_site),
                side0_i[source]:(side0_i[source] + self._options_per_site)] = \
                    np.zeros((self._options_per_site, self._options_per_site), dtype=complex)  # Delete old to term
                self.bdg_h[side0_i[dest]:(side0_i[dest] + self._options_per_site),
                start_i[source]:(start_i[source] + self._options_per_site)] = \
                    np.zeros((self._options_per_site, self._options_per_site), dtype=complex)  # Delete old from term
                self.bdg_h[side1_i[dest]:(side1_i[dest] + self._options_per_site),
                side0_i[source]:(side0_i[source] + self._options_per_site)] = to_term_0.copy()  # Stitch the gap

                # Re-arrange hopping term from neighbor 1 to gap
                to_term_1 = self.bdg_h[start_i[dest]:(start_i[dest] + self._options_per_site),
                            side1_i[source]:(side1_i[source] + self._options_per_site)].copy()  # Copy
                self.bdg_h[start_i[dest]:(start_i[dest] + self._options_per_site),
                side1_i[source]:(side1_i[source] + self._options_per_site)] = \
                    np.zeros((self._options_per_site, self._options_per_site), dtype=complex)  # Delete old to term
                self.bdg_h[side1_i[dest]:(side1_i[dest] + self._options_per_site),
                start_i[source]:(start_i[source] + self._options_per_site)] = \
                    np.zeros((self._options_per_site, self._options_per_site), dtype=complex)  # Delete old from term
                self.bdg_h[side0_i[dest]:(side0_i[dest] + self._options_per_site),
                side1_i[source]:(side1_i[source] + self._options_per_site)] = to_term_1.copy()  # Stitch the gap
        print("Number of sites removed: " + str(self.removed_site_count))
        self._site_costs.append(site_cost)

    def plot_spectrum(self, lazy=False):
        """
        Plots the spectrum of the eigenvalues calculated so far.

        :param lazy: Boolean. If True, the plot has no x variation.
        """
        if self._eigen_up_to_date is False:
            print("Eigenvalues have not been computed.")
            return
        # Sort the eigenvalues in ascending order
        spectrum = np.sort(self.eigenvalues)
        # # Remove the lines at the dislocations
        # indices = list()
        # for cost in self._site_costs:
        #     indices.append(np.searchsorted(spectrum))
        # spectrum = self.eigenvalues  # [np.where(self.eigenvalues not in self._site_costs)]  # TODO: FIX
        # Do it lazy
        if lazy:
            xs = np.zeros(len(spectrum))
        else:
            xs = np.arange(len(spectrum))
            # zero = np.searchsorted(spectrum, 0)
            # xs = xs - zero
        plt.plot(xs, spectrum, ".")
        plt.axhline(0, color="lightgrey", linestyle="--")
        plt.show()

        # # Do it fancy
        # indices = np.argsort(self.eigenvalues.real)
        # vals = self.eigenvalues[indices]
        # pos = np.where(vals >= 0)[0][0]  # Find the first positive index.


def main():
    """Runs a test"""
    # Initialize the system
    L = 3
    dofs = {"sub": ("A", "B")}
    latt = Lattice(L, dofs, ribbon_dir="y")

    # # Add an on-site term
    latt.add_diagonal(-2, 3, {"sub": 1})

    # Add a hopping term
    latt.add_hopping(1j, 0, {"sub": 0}, 1)
    # latt.dislocation(1, 1, 0, site_cost=10)

    latt.calculate_eigenvalues()

    latt.plot_spectrum()
    # latt.plot_eigenvector(10, color=True)


def main2():
    latt = Lattice(2, {"spin": ("up", "down")})
    latt2 = Lattice(2, {"spin": ("up", "down")})

    for i in range(4):
        for j in range(4):
            latt.add_hopping_old(5j, i, {"spin": 0}, j, verbose=True)
            latt2.add_hopping(5j, i, {"spin": 0}, j, verbose=True)

            np.testing.assert_almost_equal(latt.bdg_h, latt2.bdg_h)


if __name__ == "__main__":
    main()
