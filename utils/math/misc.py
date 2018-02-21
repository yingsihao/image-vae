import numpy as np


def softmax(x):
    if isinstance(x, (list, tuple)):
        return softmax(np.array(x)).tolist()

    assert isinstance(x, np.ndarray)
    return np.exp(x) / np.sum(np.exp(x), axis = 0)


def sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy):
    x = np.random.multivariate_normal([mu_x, mu_y], [
        [sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],
        [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]
    ], 1)
    return x[0][0], x[0][1]
