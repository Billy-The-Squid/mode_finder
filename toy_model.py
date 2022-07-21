from lattice import *
# try:
#     from mode_finder.lattice import *
# except ModuleNotFoundError:
#     pass
from time import perf_counter


def setup_system(L, epsilon_0, m_0, m_1, v, delta, verbose=False):
    """
    Sets up the system and returns the lattice.Lattice
    :param L:
    :param epsilon_0:
    :param m_0:
    :param m_1:
    :param v:
    :param delta:
    :param verbose:
    :return:
    """
    # Create the matrix
    total_init = perf_counter()
    latt = Lattice(L, {"s": ("up", "down"), "rho": ("s", "p-")})
    if verbose:
        print("Time to create matrix: " + str(perf_counter() - total_init))

    # Add terms
    # 1
    latt.add_diagonal(epsilon_0, 3, {"s": 0, "rho": 0}, verbose=verbose)
    # 2
    latt.add_diagonal(m_0, 3, {"s": 0, "rho": 3}, verbose=verbose)
    latt.add_hopping(0.5 * m_1, 3, {"s": 0, "rho": 3}, 0, verbose=verbose)
    latt.add_hopping(0.5 * m_1, 3, {"s": 0, "rho": 3}, 1, verbose=verbose)
    # 3
    latt.add_hopping(-1j * v / 2, 0, {"s": 3, "rho": 1}, 0, verbose=verbose)
    # 4
    latt.add_hopping(-1j * v / 2, 3, {"s": 0, "rho": 2}, 1, verbose=verbose)
    # 5
    latt.add_hopping(-1j * delta / 2, 1, {"s": 3, "rho": 0}, 0, verbose=verbose)
    latt.add_hopping(1j * delta / 2, 1, {"s": 3, "rho": 0}, 2, verbose=verbose)
    # 6
    latt.add_hopping(-1j * delta / 2, 2, {"s": 0, "rho": 3}, 1, verbose=verbose)
    latt.add_hopping(1j * delta / 2, 2, {"s": 0, "rho": 3}, 3, verbose=verbose)
    # print("Warning: term 6 has been disabled")

    if verbose:
        print("Time to create entire system: " + str(perf_counter() - total_init))

    return latt


def setup_system_old(L, epsilon_0, m_0, m_1, v, delta, verbose=False):
    """
    Sets up the system and returns the lattice.Lattice
    :param L:
    :param epsilon_0:
    :param m_0:
    :param m_1:
    :param v:
    :param delta:
    :param verbose:
    :return:
    """
    # Create the matrix
    total_init = perf_counter()
    latt = Lattice(L, {"s": ("up", "down"), "rho": ("s", "p-")})
    if verbose:
        print("Time to create matrix: " + str(perf_counter() - total_init))

    # Add terms
    # 1
    latt.add_diagonal_old(epsilon_0, 3, {"s": 0, "rho": 0}, verbose=verbose)
    # 2
    latt.add_diagonal_old(m_0, 3, {"s": 0, "rho": 3}, verbose=verbose)
    latt.add_hopping_old(0.5 * m_1, 3, {"s": 0, "rho": 3}, 0, verbose=verbose)
    latt.add_hopping_old(0.5 * m_1, 3, {"s": 0, "rho": 3}, 1, verbose=verbose)
    # 3
    latt.add_hopping_old(-1j * v / 2, 0, {"s": 3, "rho": 1}, 0, verbose=verbose)
    # 4
    latt.add_hopping_old(-1j * v / 2, 3, {"s": 0, "rho": 2}, 1, verbose=verbose)
    # 5
    latt.add_hopping_old(-1j * delta / 2, 1, {"s": 3, "rho": 0}, 0, verbose=verbose)
    latt.add_hopping_old(1j * delta / 2, 1, {"s": 3, "rho": 0}, 2, verbose=verbose)
    # 6
    latt.add_hopping_old(-1j * delta / 2, 2, {"s": 0, "rho": 3}, 1, verbose=verbose)
    latt.add_hopping_old(1j * delta / 2, 2, {"s": 0, "rho": 3}, 3, verbose=verbose)

    if verbose:
        print("Time to create entire system: " + str(perf_counter() - total_init))

    return latt


def main():
    # Initial parameters
    L = 3
    epsilon_0 = 0.3
    m_0 = 1.5
    m_1 = 0.7
    v = .1
    delta = .1

    latt = setup_system(L, epsilon_0, m_0, m_1, v, delta)

    # latt.dislocation(7, 6, 1)

    print(latt.get_block(0))

    # eigenvalues, eigenvectors = np.linalg.eigh(latt.bdg_h)
    # plt.plot(np.arange(len(eigenvalues)), eigenvalues, "o")
    # plt.show()

    latt.plot_eigenvector(0)


if __name__ == "__main__":
    main()
