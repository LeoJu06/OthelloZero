


def iter_duration(time_per_move, n_games=500, num_workers=23):


    return (60*(n_games/num_workers)*time_per_move) / 60


if __name__ == "__main__":

    time_per_move = 1.8
    number_of_games = 500
    workers = 23

    duration = iter_duration(time_per_move=time_per_move, n_games=number_of_games, num_workers=workers)

    print(f"With your given params, you need [{duration:.1f} min] for one iter")