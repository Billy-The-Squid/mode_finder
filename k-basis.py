from lattice import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as tck


L = 50
epsilon_0 = 0.3
m_0 = 1.5
m_1 = -0.7
v = 0.1
delta = 0.1


def make_h(k_x, k_y):
    h_k = np.zeros((8, 8), dtype=complex)
    h_k += epsilon_0 * kron([paulis[3], paulis[0], paulis[0]])
    h_k += (m_0 + m_1 * (np.cos(k_x) + np.cos(k_y))) * kron([paulis[3], paulis[0], paulis[3]])
    h_k += v * np.sin(k_x) * kron([paulis[0], paulis[3], paulis[1]])
    h_k += v * np.sin(k_y) * kron([paulis[3], paulis[0], paulis[2]])
    h_k += delta * np.sin(k_x) * kron([paulis[1], paulis[3], paulis[0]])
    h_k += delta * np.sin(k_y) * kron([paulis[2], paulis[0], paulis[3]])
    np.testing.assert_almost_equal(h_k.conj().transpose(), h_k)
    return h_k


def main():
    # Iterate through the k vals
    delta_k = 2 * np.pi / L
    points = []
    all_evals = []
    normal_state_only = False  # Only matters when epsilon != 0 or delta != 0
    for n in range(L):
        # for m in range(L):  # TODO: CHANGE BACK
        k_x = n * delta_k
        if k_x > np.pi:
            k_x -= 2 * np.pi
        k_y = 0  # m * delta_k  # TODO: CHANGE BACK
        if k_y > np.pi:
            k_y -= 2 * np.pi
        h_k = make_h(k_x, k_y)
        if normal_state_only:
            e_vals = np.linalg.eigvals(h_k[0:4, 0:4])
        else:
            e_vals = np.linalg.eigvals(h_k)
        for val in e_vals:
            points.append(np.array([k_x, k_y, val]))
            all_evals.append(val)

    # # Plot values sorted
    # all_evals.sort()
    # all_evals = np.array(all_evals)
    # plt.plot(np.arange(len(all_evals)), all_evals, "o")
    # plt.show()

    # Plot bands
    plot_points = np.vstack(points)
    plt.plot(plot_points[:, 0]/np.pi, plot_points[:, 2], ".")
    plt.title("Band structure (%dx%d lattice)" %(L, L))
    plt.xlabel("Crystal momentum $k_x$")
    plt.ylabel("Energy eigenvalue")
    axes = plt.figure(num=1).get_axes()[0]
    axes.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\\frac{\pi}{a}$'))
    axes.xaxis.set_major_locator(tck.MultipleLocator(base=1/2))
    plt.show()

    # # Try to plot 3D
    # plot_points = np.vstack(points)
    # axes = plt.axes(projection="3d")
    # axes.scatter(plot_points[:, 0], plot_points[:, 1], plot_points[:, 2])
    # plt.show()

    print("Kappa: " + str(kappa()))


def dirac_point():
    """Not very intelligent"""
    # Iterate through the k vals
    delta_k = 2 * np.pi / L
    points = []
    normal_state_only = True
    for n in range(15, 35):
        k_x = 0  # n * delta_k
        if k_x > np.pi:
            k_x -= 2 * np.pi
        k_y = n * delta_k
        if k_y > np.pi:
            k_y -= 2 * np.pi
        h_k = make_h(k_x, k_y)
        if normal_state_only:
            e_vals = np.linalg.eigvals(h_k[0:4, 0:4])
        else:
            e_vals = np.linalg.eigvals(h_k)
        for val in e_vals:
            points.append(np.array([k_y, val]))

    # Plot bands
    plot_points = np.vstack(points)
    plt.plot(plot_points[:, 0], plot_points[:, 1], ".")
    plt.show()


def kappa():
    inversion = kron([paulis[3], paulis[0], paulis[3]])
    kappa = 0
    # Iterate over the points
    for x, y in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        # print("(%d, %d)" %(x, y))
        k_x = x * np.pi
        k_y = y * np.pi
        h_k = make_h(k_x, k_y)

        vals, vecs = np.linalg.eigh(h_k)
        # indices = np.argsort(vals)
        # vals = vals[indices]
        # vecs = vecs[:, indices]

        # Iterate over the low-eigenvalue bands
        for i in range(4):
            vec = vecs[:, i]
            inverted = np.dot(inversion, vec)
            contribution = np.vdot(vec, inverted)
            kappa += contribution
    kappa /= 4
    # if kappa < 0:
    #     kappa += 4
    return kappa


if __name__ == "__main__":
    main()
