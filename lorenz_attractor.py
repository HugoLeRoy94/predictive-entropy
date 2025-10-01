
import numpy as np
from scipy.integrate import solve_ivp

def lorenz(t, xyz, sigma=10, rho=28, beta=8/3):
    """Lorenz system of ordinary differential equations."""
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def simulate_lorenz(t_span, initial_conditions, dt):
    """
    Simulates the Lorenz attractor.

    Args:
        t_span (tuple): Start and end times for the simulation (t_start, t_end).
        initial_conditions (list or np.ndarray): Initial values for [x, y, z].
        dt (float): Time step for the output.

    Returns:
        tuple: A tuple containing:
            - t (np.ndarray): Time points.
            - xyz (np.ndarray): Array of shape (n_points, 3) with the trajectory.
    """
    sol = solve_ivp(lorenz, t_span, initial_conditions, dense_output=True)
    t = np.arange(t_span[0], t_span[1], dt)
    xyz = sol.sol(t).T
    return t, xyz
