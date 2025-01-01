import numpy as np


def add_dirichlet_noise(priors, alpha, epsilon):
    """
    Adds Dirichlet noise to the prior probabilities to encourage exploration.

    Args:
        priors (list or np.ndarray): The original prior probabilities for each move (sum should be 1).
        alpha (float): The Dirichlet distribution parameter (e.g., 0.03 for a small, focused noise).
        epsilon (float): The weighting factor for combining noise with priors (e.g., 0.25).

    Returns:
        np.ndarray: The priors with Dirichlet noise added.
    """
    # Generate Dirichlet noise
    dirichlet_noise = np.random.dirichlet([alpha] * len(priors))

    # Combine the original priors with the noise
    noisy_priors = (1 - epsilon) * np.array(priors) + epsilon * dirichlet_noise

    return noisy_priors
