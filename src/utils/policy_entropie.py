
import numpy as np


def relative_policy_entropy(policy_probs):
    policy_probs = np.array(policy_probs)
    policy_probs = policy_probs[policy_probs > 0]  # log(0) verhindern
    H = -np.sum(policy_probs * np.log(policy_probs))
    H_max = np.log(len(policy_probs))
    return H / H_max if H_max > 0 else 0
