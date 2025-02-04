import torch
import torch.multiprocessing as mp
import queue
import time
import numpy as np
from tqdm import tqdm
from src.neural_net.model import OthelloZeroModel
from src.othello.othello_game import OthelloGame
from src.config.hyperparameters import Hyperparameters
from src.mcts.worker import Worker

class Manager:
    def __init__(self, model, state_shape, num_workers):
        self.model = model
        self.state_shape = state_shape
        self.num_workers = num_workers

        # Shared memory for game states
        self.shared_states = torch.zeros((num_workers, *state_shape), dtype=torch.float32)
        self.shared_states.share_memory_()

        # Shared memory for responses
        self.shared_policies = torch.zeros((num_workers, 65))  # Assuming 64 possible actions
        self.shared_values = torch.zeros((num_workers,))
        self.shared_policies.share_memory_()
        self.shared_values.share_memory_()

        # Communication queue for requests
        self.request_queue = mp.Queue()

        # Sync events for worker-manager communication
        self.response_events = [mp.Event() for _ in range(num_workers)]

    def manage_workers(self, timeout=0.001):
        while True:
            batch_indices, worker_ids = [], []
            start_time = time.time()

            # Collect batch requests
            while len(batch_indices) < self.num_workers:
                try:
                    worker_id, state_index = self.request_queue.get_nowait()
                    batch_indices.append(state_index)
                    worker_ids.append(worker_id)
                except queue.Empty:
                    if batch_indices or (time.time() - start_time) >= timeout:
                        break

            # Process batch if any requests exist
            if batch_indices:
                batch_states = self.shared_states[batch_indices]  

                with torch.no_grad():
                    policies, values = self.model.predict_batch(batch_states)

                print(f"Manager processing {len(batch_states)} requests", end="\r")

                # Store results in shared memory and notify workers
                for worker_id, policy, value in zip(worker_ids, policies, values):
                    self.shared_policies[worker_id] = policy
                    self.shared_values[worker_id] = value
                    self.response_events[worker_id].set()  # Notify worker

def worker_process_function(worker_id, manager):
    """Worker process function that runs MCTS simulations."""
    shared_states = manager.shared_states  # Access shared memory
    
    worker_mcts = Worker(
        worker_id=worker_id,
        request_queue=manager.request_queue,
        shared_states=shared_states,
        shared_policies=manager.shared_policies,
        shared_values=manager.shared_values,
        response_event=manager.response_events[worker_id],
    )

    state = OthelloGame().get_init_board()
    to_play = -1  # Example: Player -1 starts
    start_time = time.time()
    
    for i in tqdm(range(60), desc=f"Worker {worker_id}"):
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
    num_workers = 12  # Adjust based on CPU cores

    print(f"Using {num_workers} workers.")
    mp.set_sharing_strategy('file_system')

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
