import numpy as np
from scipy.sparse import linalg as spla
import tabulate


def gmres(A, b, x0, max_iter, tol):
    k = max_iter

    # Initialize matrices Q and H
    Q = np.empty((len(b), k + 1))
    H = np.zeros((k + 1, k))

    # Calculate the initial residual vector r0 = b - A(x0)
    r0 = b - A @ x0
    beta = np.linalg.norm(r0)
    Q[:, 0] = r0 / beta

    # Initialize the first standard basis vector e1 of length k + 1
    e1 = np.zeros(k + 1)
    e1[0] = 1

    # Perform the Arnoldi iteration
    for j in range(k):
        Q[:, j + 1] = A @ Q[:, j]

        # Update H
        for i in range(j + 1):
            H[i, j] = Q[:, i].T @ Q[:, j + 1]
            Q[:, j + 1] -= H[i, j] * Q[:, i]

        H[j + 1, j] = np.linalg.norm(Q[:, j + 1])

        # Avoid dividing by zero
        if H[j + 1, j] > tol:
            Q[:, j + 1] /= (H[j + 1, j] + 1e-13)

        # Solve the least squares problem to find y
        y = np.linalg.lstsq(H[:j + 2, :j + 1], beta *
                            e1[:j + 2], rcond=None)[0]
        x = Q[:, :j + 1] @ y + x0
        res = np.linalg.norm(A @ x - b)

        if res < tol:
            break

    return x, res


if __name__ == "__main__":
    n_vals = [10, 50, 100, 250, 500, 1000]
    max_iter = 1000
    tol = 1e-10
    errors = []

    for n in n_vals:
        A = np.random.rand(n, n)
        b = np.random.rand(n)
        x, res = gmres(A, b, np.zeros_like(b), max_iter, tol)
        x_, info = spla.gmres(A, b, tol=tol, restart=max_iter)
        e = np.linalg.norm(x - x_) / np.linalg.norm(x_)
        errors.append(e)

    # Tabulate n, errors
    table = tabulate.tabulate(zip(n_vals, errors), headers=[
                              "n", "relative error in x"])
    print(table)
