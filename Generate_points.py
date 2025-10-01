import numpy as np

def gaussian_radial_points(N, d, center=None, sigma=1.0, mu=0.0, rng=None):
    """
    Generate N random points in R^d around a center, with the distance
    from center distributed as N(mu, sigma^2).
    
    Parameters
    ----------
    N : int
        Number of points.
    d : int
        Dimension of the space.
    center : array_like, shape (d,), optional
        Center point. Defaults to origin.
    sigma : float
        Standard deviation of the Gaussian distance.
    mu : float
        Mean of the Gaussian distance.
    rng : np.random.Generator or None
        Random number generator.
    
    Returns
    -------
    pts : ndarray, shape (N, d)
        Sampled points.
    """
    if rng is None:
        rng = np.random.default_rng()
    if center is None:
        center = np.zeros(d)
    center = np.asarray(center, dtype=float)

    # Sample Gaussian radii (can be negative, take absolute if you want nonnegative distances only)
    r = rng.normal(loc=mu, scale=sigma, size=N)

    # Sample random directions uniformly on the sphere
    dirs = rng.normal(size=(N, d))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]

    pts = center + r[:, None] * dirs
    return pts
