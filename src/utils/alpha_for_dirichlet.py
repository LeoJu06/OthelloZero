


def alpha_for_dirichlet(n_valid_moves):

    if n_valid_moves > 0: 
        return min(1, 10/n_valid_moves)
    else:
        return 10
    

