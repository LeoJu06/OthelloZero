


def iter_duration(time_per_move, n_games=500, num_workers=23):

    max_length = 60

    game_length = time_per_move * max_length

    games_per_worker = n_games / num_workers

    return (games_per_worker * game_length) / 60



def report(params:dict):


    duration = iter_duration(params.get("TIME_PER_MOVE"), params.get("NUMBER_OF_GAMES"), params.get("WORKERS"))

    print("__REPORT_TRAINING__")
    for description, figure in params.items():

        print(f"{description:20} => {figure}")

  
    print(f"{'EPISODE_DUARATION':20} => {duration:.1f}min")
    



if __name__ == "__main__":

    time_per_move = 1.2
    games_per_worker = 8
    
    workers = 23
    number_of_games = workers * games_per_worker
    amount_iterations = 10000

    params = {"TIME_PER_MOVE" : time_per_move,
              "GAMES_PER_WORKER": games_per_worker,
               "WORKERS": workers,
                "NUMBER_OF_GAMES": number_of_games,
                "TRAINING_EXAMPLES": number_of_games*4*60,
                "AMMOUNT_ITERATIONS":amount_iterations}
    
    report(params)



   



   