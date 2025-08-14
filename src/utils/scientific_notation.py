from math import log10

def scientific_notation(legal_moves, branching_factor):
    

    value = legal_moves ** branching_factor
    exp = int(log10(value))
    coeff = value / (10 ** exp)

    return f"{coeff:.2f} · 10^{exp}"

def scientific_e_format(base, exponent):
    value = base ** exponent
    return f"{value:.1e}"

def compute_efficiency_e_format(branching_factor, depth, actual_nodes):
    total_nodes = branching_factor ** depth
    efficiency = total_nodes / actual_nodes
    return f"{efficiency:.1e}"


