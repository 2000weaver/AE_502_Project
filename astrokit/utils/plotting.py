import matplotlib.pyplot as plt

def plot_trajectory_3d(states):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.plot(states[0], states[1], states[2])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.show()


def plot_jacobi(t, jacobi):
    plt.plot(t, jacobi)
    plt.xlabel("Time")
    plt.ylabel("Jacobi Constant")
    plt.show()