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
    def __init__(self, L: int, dofs: dict = None):
        """
        Creates an empty matrix representing the BdG Hamiltonian (in r-basis) for the system.
        We assume the system is a square 2D lattice.

        :param L: The length of one side of the lattice, or the number of sites along one axis.
        :param dofs: The degrees of freedom at each site. This should be a dictionary with dof names
            (as strings) mapping to lists of possible values. The dictionary should not be empty and
            each list should have more than one entry.
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
        self.sparse_h = None
        self._site_costs = []

    def get_index(self, x: int, y: int, dofs: dict, particle=True):
        """
        Returns the integer index of the specified site with the specified degree of freedom. (x,y) = (0,0) is in a
            corner.
        :param x: The x coordinate of the site.
        :param y: The y coordinate of the site.
        :param dofs: A dictionary mapping the strings labeling the dofs to the index of their value in the dof list.
        :param particle:
        :return: The integer index of the site specified.
        """
        if x >= self.L or y >= self.L or x < 0 or y < 0:
            raise IndexError("x, y indices exceed size of array")
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

    def add_diagonal_old(self, term, pauli_index, pauli_indices, verbose=False):
        """
        Adds a c_x c_x term to the matrix.
        :param term: The constant
        :param pauli_index: 0, 1, 2, or 3: specifies the Pauli matrix to be used for particle-hole interactions. [BETTER
            DOCS!] 0 uses the identity.
        :param pauli_indices: A dictionary with the same keys as the dofs attributes, each pointing to either 0, 1, 2,
            or 3, specifying the matrix associated with this degree of freedom.
        """
        start_t = perf_counter()
        on_site_term = kron([paulis[pauli_indices[dof]] for dof in self.dofs])
        full = term * kron([paulis[pauli_index], np.identity(self.L * self.L), on_site_term])
        self.bdg_h += full
        self._eigen_up_to_date = False
        np.testing.assert_almost_equal(self.bdg_h.conj().transpose(), self.bdg_h)
        if verbose:
            print("Time to add diagonal: " + str(perf_counter() - start_t))

    def add_diagonal(self, term, pauli_index, pauli_indices, verbose=False):
        """

        :param term:
        :param pauli_index:
        :param pauli_indices:
        :return:
        """
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

    def add_hopping_old(self, term, pauli_index, pauli_indices, direction, verbose=False):
        """
        Adds a c_x c_{x+-1} term to the matrix. Preserves Hermiticity but will not add the opposite-direction hopping
            term if pauli_index is 1 or 2.
        :param term:
        :param pauli_index:
        :param pauli_indices:
        :param direction: 0 for x->x+1, 1 for y->y+1, 2 for x->x-1, 3 for y->y-1
        :return: None. Modifies bdg_h directly.
        """
        start_t = perf_counter()
        ad_paulis = [
            np.array([[1, 0], [0, 1]]),
            np.array([[0, 1], [0, 0]]),
            np.array([[0, -1j], [0, 0]]),
            np.array([[1, 0], [0, -1]])
        ]
        on_site_term = kron([paulis[pauli_indices[dof]] for dof in self.dofs])
        di_1 = term * np.ones(self.L - 1)
        if direction == 0:
            y_mat = np.identity(self.L)
            x_mat = np.diag(di_1, -1)
        elif direction == 1:
            x_mat = np.identity(self.L)
            y_mat = np.diag(di_1, -1)
        if direction == 2:
            y_mat = np.identity(self.L)
            x_mat = np.diag(di_1, 1)
        elif direction == 3:
            x_mat = np.identity(self.L)
            y_mat = np.diag(di_1, 1)
        half = kron([ad_paulis[pauli_index], y_mat, x_mat, on_site_term])
        self.bdg_h += half + half.conj().transpose()
        self._eigen_up_to_date = False
        np.testing.assert_almost_equal(self.bdg_h.conj().transpose(), self.bdg_h)
        if verbose:
            print("Time to add diagonal: " + str(perf_counter() - start_t))

    def add_hopping(self, term, pauli_index, pauli_indices, direction, verbose=False):
        """

        :param term:
        :param pauli_index:
        :param pauli_indices:
        :param direction:
        :param verbose:
        :return:
        """
        start_t = perf_counter()
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
                site_i = [
                    self.get_index(x, y, init, particle=True),
                    self.get_index(x, y, init, particle=False)
                ]
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
        Returns a block of the matrix indicated by the index. 0 gives the c*c terms, 1 the c*c* terms, 2 the cc terms,
            and 3 the cc* terms.
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
        greatest, stored in lattice.eigenvalues and lattice.eigenvectors.
        :return:
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

        """
        if count >= self.bdg_h.shape[0] - 1:
            self.calculate_eigenvalues()
            return
        init = perf_counter()
        if self._eigen_up_to_date is False or self._eigen_up_to_date < count:
            if self.sparse_h is None:
                self.sparse_h = spr.csr_matrix(self.bdg_h)
            if verbose:
                print("Time to generate sparse matrix: " + str(perf_counter() - init))
                next = perf_counter()
            vals, vecs = eigsh(self.sparse_h, k=count, sigma=0, which="LM")
            if verbose:
                print("Time to find some eigenvalues: " + str(perf_counter() - next))
            indices = np.argsort(vals ** 2)
            self.eigenvalues = vals[indices]
            self.eigenvectors = vecs[:, indices]
            self._eigen_up_to_date = count
        if verbose:
            print("Total time to calculate some eigenvalues: " + str(perf_counter() - init))

    def plot_eigenvector(self, index, color=False):
        """
        Plots the probability density of the wavefunction indicated by the eigenvector at the specified index.
        :param index: The index of the eigenvector in lattice.eigenvectors.
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
        plt.title("Spatial probability distribution (%dx%d lattice)" %(self.L, self.L))
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.colorbar()
        plt.show()

    def dislocation(self, x_start, y_start, direc, site_cost=10):
        """
        Creates a dislocation starting at the specified point and continuing in the specified direction by removing the
            sites from that site and continuing in that direction. The sites on either side of the missing sites will
            have hopping terms to each other.
        This should only be used once in a lattice, and not before adding additional hopping or on-site terms.
        :param x_start:
        :param y_start:
        :param direc: 0 for +x, 1 for +y, 2 for -x, 3 for -y
        """
        # NOTE: This code will NOT generalize if next-nearest-neighbor terms are introduced.
        self._eigen_up_to_date = False

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
        plt.axhline(0, color="lightgrey")
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
    latt = Lattice(L, dofs)

    # # Add an on-site term
    latt.add_diagonal(-2, 3, {"sub": 1})

    # Add a hopping term
    latt.add_hopping(1j, 0, {"sub": 0}, 0)
    latt.dislocation(1, 1, 0, site_cost=10)

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
