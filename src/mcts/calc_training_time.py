


def iter_duration(time_per_move, n_games=120, num_workers=23):


    return (60*(n_games/num_workers)*time_per_move) / 60


if __name__ == "__main__":

    time_per_move = 1.85

    duration = iter_duration(time_per_move=time_per_move)

    print(duration)