
import numpy as np
from scipy.spatial.distance import pdist, squareform

def time_delay_embedding(series, m, tau):
    """
    Creates a time-delay embedding of a time series.

    Args:
        series (np.ndarray): The input time series (1D array).
        m (int): The embedding dimension.
        tau (int): The time delay.

    Returns:
        np.ndarray: The embedded time series (n_points - (m-1)*tau, m).
    """
    n = len(series)
    embedded_series = np.zeros((n - (m - 1) * tau, m))
    for i in range(m):
        embedded_series[:, i] = series[i * tau : i * tau + len(embedded_series)]
    return embedded_series

def correlation_integral(embedded_series, epsilon):
    """
    Calculates the correlation integral C(epsilon) for a given epsilon.

    Args:
        embedded_series (np.ndarray): The embedded time series.
        epsilon (float): The radius.

    Returns:
        float: The correlation integral.
    """
    dist_matrix = squareform(pdist(embedded_series, 'chebyshev'))
    n = len(embedded_series)
    # Heaviside function
    C = np.sum(dist_matrix < epsilon) / (n * (n - 1))
    return C

def cohen_procaccia_entropy(series, m, tau, epsilon):
    """
    Calculates the Cohen-Procaccia entropy for a given epsilon.

    Args:
        series (np.ndarray): The input time series (1D array).
        m (int): The embedding dimension.
        tau (int): The time delay.
        epsilon (float): The radius.

    Returns:
        float: The Cohen-Procaccia entropy h(epsilon, tau).
    """
    # Correlation integral for dimension m
    embedded_m = time_delay_embedding(series, m, tau)
    C_m = correlation_integral(embedded_m, epsilon)

    # Correlation integral for dimension m+1
    embedded_m1 = time_delay_embedding(series, m + 1, tau)
    C_m1 = correlation_integral(embedded_m1, epsilon)

    if C_m == 0 or C_m1 == 0:
        return 0.0

    # The paper uses the natural logarithm
    h = (1 / tau) * (np.log(C_m) - np.log(C_m1))
    return h
