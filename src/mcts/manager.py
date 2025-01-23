import numpy as np
from src.neural_net.model import OthelloZeroModel
from src.othello.othello_game import OthelloGame
from src.config.hyperparameters import Hyperparameters
from src.mcts.worker import Worker
import multiprocessing as mp


class Manager:
    def __init__(self, model: OthelloZeroModel):
        """
        Initialize the manager with batching and coordination logic.
        """
        self.model = model

    def manage_workers(self, request_queue, response_queues, batch_size=10):
        while True:
            batch, worker_ids = [], []
            while len(batch) < batch_size and not request_queue.empty():
                worker_id, state = request_queue.get()
                batch.append(state)
                worker_ids.append(worker_id)

            if batch:
                policies, values = self.model.predict_batch(batch)
                for worker_id, policy, value in zip(worker_ids, policies, values):
                    response_queues[worker_id].put(
                        {"worker_id": worker_id, "policy": policy, "value": value}
                    )


def worker_process_function(worker_id, request_queue, response_queue, state, to_play):
    worker_mcts = Worker(worker_id=worker_id, request_queue=request_queue, response_queue=response_queue)
    worker_mcts.run(state, to_play)


if __name__ == "__main__":
    device = Hyperparameters.Neural_Network["device"]
    game = OthelloGame()
    model = OthelloZeroModel(game.rows, game.get_action_size(), device)
    model.eval()

    manager = Manager(model=model)
    mp.set_start_method("spawn")

    num_workers = 10 # int(mp.cpu_count() * 0.85)
    print(f"Num Workers are calculated by int(num_cores*0.85) => {num_workers}")

    request_queue = mp.Queue()
    response_queues = [mp.Queue() for _ in range(num_workers)]

    manager_process = mp.Process(
        target=manager.manage_workers, args=(request_queue, response_queues)
    )
    manager_process.start()

    workers = []
    for i in range(num_workers):
        state = np.copy(game.get_init_board())
        worker_process = mp.Process(
            target=worker_process_function,
            args=(i, request_queue, response_queues[i], state, -1),
        )
        workers.append(worker_process)
        worker_process.start()

    for worker_process in workers:
        worker_process.join()

    manager_process.terminate()
