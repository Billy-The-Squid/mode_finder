import matplotlib.pyplot as plt
import numpy as np

from toy_model import *
from k_basis import *


def scan_m0_m1():
    epsilon_0 = 0.5
    v = 0.4
    delta = 0.15

    m_0s = np.arange(0.2, 2.2, 0.2)
    m_1s = np.arange(0.2, 2.2, 0.2)
    m_0s, m_1s = np.meshgrid(m_0s, m_1s)

    for j in range(m_1s.shape[1]):
        for i in range(m_0s.shape[0]):
            print("Working on m_0 = %.2f, m_1 = %.2f" %(m_0s[i, j], m_1s[i, j]))
            kappa_1 = kappa(new_values={
                "epsilon_0": epsilon_0,
                "m_0": m_0s[i, j],
                "m_1": m_1s[i, j],
                "v": v,
                "delta": delta
            }).real
            kappa_neg = kappa(new_values={
                "epsilon_0": epsilon_0,
                "m_0": -m_0s[i, j],
                "m_1": m_1s[i, j],
                "v": v,
                "delta": delta
            }).real

            latt = setup_system(50, epsilon_0, m_0s[i, j], m_1s[i, j], v, delta)
            latt_neg = setup_system(50, epsilon_0, -m_0s[i, j], m_1s[i, j], v, delta)
            latt.dislocation(30, 25, 0)
            latt_neg.dislocation(30, 25, 0)
            latt.calculate_eigenvalues_sparse(2)
            latt_neg.calculate_eigenvalues_sparse(2)

            print("Kappa: " + str(round(kappa_1)))
            latt.plot_eigenvector(0, color=True)
            latt_neg.plot_eigenvector(0, color=True)

            # spectrum = np.sort(latt.eigenvalues)
            # spectrum_neg = np.sort(latt_neg.eigenvalues)
            # xs = np.arange(len(spectrum))
            #
            # plt.plot(xs, spectrum, "o")
            # plt.plot(xs, spectrum_neg, "^r")
            # plt.title("$\kappa = %.2f, \kappa_- = %.2f$" %(kappa_1, kappa_neg))
            # plt.axhline(0, color="lightgrey", linestyle="--")
            # plt.show()


def scan_lattice_size(epsilon_0, m_0, m_1, v, delta, max_len = 7):
    latts = list((0,))
    eigenvalues = list()
    for i in range(1, max_len):
        print("Starting: %dx%d" %(10 * i, 10 * i))
        latts.append(setup_system(10 * i, epsilon_0, m_0, m_1, v, delta))
        latts[i].dislocation(6 * i, 5 * i, 0)
        latts[i].calculate_eigenvalues_sparse(10, verbose=True)
        eigenvalues.append(abs(latts[i].eigenvalues[0]))

    eigenvalues = np.array(eigenvalues)
    ln_eigs = np.log10(eigenvalues)
    xs = 10 * np.arange(1, max_len)

    # plt.plot(xs, eigenvalues, "o")
    # plt.xlabel("Lattice side length $N$")
    # plt.ylabel("Energy eigenvalue")
    # plt.show()

    plt.plot(xs, ln_eigs, "o")
    plt.xlabel("Lattice side length $N$")
    plt.ylabel("Energy eigenvalue $\log_{10}$")
    plt.show()

    for i in range(1, max_len):
        latts[i].plot_eigenvector(0, color=True)


if __name__ == "__main__":
    scan_lattice_size(0.5, 1, 0.8, 0.4, 0.15)
