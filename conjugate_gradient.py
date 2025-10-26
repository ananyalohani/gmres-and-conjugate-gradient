import numpy as np
from scipy.sparse import linalg as spla
import tabulate
from sklearn import datasets as skd


def conjugate_gradient(A, b, x, tol, max_iter):
    r = b - A @ x
    p = r
    prev_norm = r.T @ r

    for _ in range(max_iter):
        Ap = A @ p
        alpha = prev_norm / (p.T @ Ap)
        x += alpha * p
        r -= alpha * Ap
        norm = r.T @ r
        if np.sqrt(norm) < tol:
            break
        p = r + (norm / prev_norm) * p
        prev_norm = norm

    return x


if __name__ == "__main__":
    n_vals = [10, 50, 100, 250, 500, 1000]
    max_iter = 1000
    tol = 1e-10
    errors = []

    for n in n_vals:
        A = skd.make_spd_matrix(n)
        b = np.random.rand(n)
        x0 = np.ones(n)
        x = conjugate_gradient(A, b, np.copy(x0), tol, max_iter)
        x_ = spla.cg(A, b, x0, tol=tol, maxiter=max_iter)[0]
        e = np.linalg.norm(x - x_) / np.linalg.norm(x_)
        errors.append(e)

    # Tabulate n, errors
    table = tabulate.tabulate(zip(n_vals, errors), headers=[
                              "n", "relative error in x"])
    print(table)
