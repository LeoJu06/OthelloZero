import torch
import torch.multiprocessing as mp
import queue
import time
import numpy as np
from tqdm import tqdm
from src.neural_net.model import OthelloZeroModel
from src.othello.othello_game import OthelloGame
from src.config.hyperparameters import Hyperparameters
from src.mcts.mcts import MultiprocessedMCTS


class Manager:
    def __init__(self, model, state_shape, num_workers):
        self.model = model
        self.state_shape = state_shape
        self.num_workers = num_workers

        # Create shared memory for game states using PyTorch
        self.shared_states = torch.zeros(
            (num_workers, *state_shape), dtype=torch.float32
        )
        self.shared_states.share_memory_()  # Enable shared memory

        # Communication queues
        self.request_queue = mp.Queue()
        self.response_queues = [mp.Queue() for _ in range(num_workers)]

    def manage_workers(self, timeout=0.001):
        batch_size = self.num_workers
        while True:
            batch_indices, worker_ids = [], []
            start_time = time.time()

            # Collect batch requests
            while len(batch_indices) < batch_size:
                try:
                    worker_id, state_index = self.request_queue.get(block=False)
                    batch_indices.append(state_index)
                    worker_ids.append(worker_id)
                except queue.Empty:
                    if batch_indices or (time.time() - start_time) >= timeout:
                        break

            # Model prediction for batch
            if batch_indices:
                batch_states = self.shared_states[
                    batch_indices
                ]  # Get batch from shared memory

                with torch.no_grad():
                    policies, values = self.model.predict_batch(batch_states)

                print(f"Manager processing {len(batch_states)} requests", end="\r")

                # Send results back to workers
                for worker_id, policy, value in zip(worker_ids, policies, values):
                    self.response_queues[worker_id].put(
                        {"worker_id": worker_id, "policy": policy, "value": value}
                    )


def worker_process_function(worker_id, manager):
    """Worker process function that runs MCTS simulations."""
    shared_states = manager.shared_states  # Access shared memory

    worker_mcts = MultiprocessedMCTS(
        idx=worker_id,
        request_queue=manager.request_queue,
        response_queue=manager.response_queues[worker_id],
        shared_states=shared_states,
    )

    state = OthelloGame().get_init_board()

    to_play = -1  # Example: Player -1 starts
    start_time = time.time()

    for i in tqdm(range(10), desc=f"Worker {worker_id}"):
        worker_mcts.run(state, to_play)

    elapsed_time = time.time() - start_time
    print(f"Worker {worker_id}: {i+1} runs completed in {elapsed_time:.4f} sec")
    print(f"Worker {worker_id}: Avg time per turn = {(elapsed_time)/(i+1):.3f} sec")


def main():
    """Main function to start the manager and worker processes."""
    device = Hyperparameters.Neural_Network["device"]
    game = OthelloGame()
    model = OthelloZeroModel(game.rows, game.get_action_size(), device)
    model.eval()

    state_shape = (game.rows, game.columns)
    num_workers = 20  # Adjust based on CPU cores

    print(f"Using {num_workers} workers.")

    mp.set_start_method("spawn", force=True)

    # Initialize Manager
    manager = Manager(model=model, state_shape=state_shape, num_workers=num_workers)

    # Start Manager Process
    manager_process = mp.Process(target=manager.manage_workers)
    manager_process.start()

    # Start Worker Processes
    workers = []
    for i in range(num_workers):
        worker_process = mp.Process(target=worker_process_function, args=(i, manager))
        workers.append(worker_process)
        worker_process.start()

    # Wait for Workers
    for worker in workers:
        worker.join()

    # Cleanup
    manager_process.terminate()


if __name__ == "__main__":
    main()
