
import numpy as np
import matplotlib.pyplot as plt
from lorenz_attractor import simulate_lorenz
from entropy import cohen_procaccia_entropy, time_delay_embedding

# --- 1. Simulate Lorenz Attractor ---
def main():
    """
    Main function to run the Lorenz simulation and entropy calculation.
    """
    # Lorenz parameters
    t_span = (0, 100)
    initial_conditions = [0, 1, 1.05]
    dt = 0.01

    # Generate Lorenz time series
    t, xyz = simulate_lorenz(t_span, initial_conditions, dt)
    series = xyz[:, 0]  # Use the x-component for entropy calculation

    # --- 2. Calculate Cohen-Procaccia Entropy ---

    # Entropy parameters
    m = 2  # Embedding dimension
    tau = 10  # Time delay

    # We calculate the entropy for a range of epsilon values
    epsilons = np.logspace(-2, 1, 20)
    entropies = []

    print(f"Calculating Cohen-Procaccia entropy for m={m}, tau={tau}")
    for epsilon in epsilons:
        h = cohen_procaccia_entropy(series, m, tau, epsilon)
        entropies.append(h)
        print(f"epsilon={epsilon:.4f}, h(epsilon, tau)={h:.4f}")

    # --- 3. Plotting (Optional) ---
    # You can uncomment this section to plot the results

    # plt.figure(figsize=(12, 5))

    # # Plot Lorenz attractor
    # plt.subplot(1, 2, 1, projection='3d')
    # plt.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    # plt.title("Lorenz Attractor")

    # # Plot entropy vs epsilon
    # plt.subplot(1, 2, 2)
    # plt.plot(np.log(epsilons), entropies, 'o-')
    # plt.xlabel("log(epsilon)")
    # plt.ylabel("h(epsilon, tau)")
    # plt.title("Cohen-Procaccia Entropy")
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    # Note: This script requires numpy, scipy, and matplotlib.
    # You can install them with: pip install numpy scipy matplotlib
    main()
