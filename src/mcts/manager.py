import torch
import torch.multiprocessing as mp
import threading
import queue
import time
import numpy as np
from tqdm import tqdm
from src.neural_net.model import OthelloZeroModel
from src.othello.othello_game import OthelloGame
from src.config.hyperparameters import Hyperparameters
from src.mcts.mcts import MultiprocessedMCTS


class Manager:
    """
    Manages the communication between worker processes and the neural network model.

    The Manager class is responsible for handling prediction requests from worker processes,
    batching these requests, and sending the results back to the workers. It uses shared memory
    to efficiently pass game states between processes.

    Attributes:
        model (OthelloZeroModel): The neural network model used for predictions.
        state_shape (tuple): The shape of the game state (rows, columns).
        num_workers (int): The number of worker processes.
        shared_states (torch.Tensor): Shared memory tensor for storing game states.
        request_queue (mp.Queue): Queue for receiving prediction requests from workers.
        response_queues (list): List of queues for sending prediction results to workers.
    """

    def __init__(self, model, state_shape, num_workers):
        """
        Initializes the Manager with the model, state shape, and number of workers.

        Args:
            model (OthelloZeroModel): The neural network model for predictions.
            state_shape (tuple): The shape of the game state (rows, columns).
            num_workers (int): The number of worker processes.
        """
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
        """
        Manages worker requests by batching them and performing predictions.

        This method continuously listens for prediction requests from workers, batches them,
        performs predictions using the neural network model, and sends the results back to the workers.

        Args:
            timeout (float): The maximum time to wait for requests before processing a batch.
        """
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

def init_manager(model:OthelloZeroModel, hyperparameters:Hyperparameters) -> Manager:
    """Creates a manager object and returns it"""

    game = OthelloGame()
    state_shape = (game.rows, game.columns)
    num_workers = hyperparameters.Coach["num_workers"]
    manager = Manager(model=model, state_shape=state_shape, num_workers=num_workers)
    return manager

def init_manager_process(manager:Manager):

    """Creates the manager process and starts it with his target function - manager_workers.
    returns the manager process"""

    #device = Hyperparameters.Neural_Network["device"]
    #model = OthelloZeroModel(game.rows, game.get_action_size(), device)
    #model.eval()

 
    # Start Manager Process
    manager_process = mp.Process(target=manager.manage_workers)
    manager_process.start()
    return manager_process

def terminate_manager_process(manager_process: mp.Process):
    if manager_process.is_alive():
        manager_process.terminate()
        manager_process.join()  # <-- Das hier ist entscheidend


def create_multiprocessed_mcts(worker_id, manager):
    shared_states = manager.shared_states  # Access shared memory

    worker_mcts = MultiprocessedMCTS(
        idx=worker_id,
        request_queue=manager.request_queue,
        response_queue=manager.response_queues[worker_id],
        shared_states=shared_states,
    )

    return worker_mcts



def worker_process_function(worker_id, manager):
    """
    Worker process function that runs MCTS simulations.

    Each worker process runs MCTS simulations independently, requesting predictions
    from the Manager when needed and using the results to guide the search.

    Args:
        worker_id (int): The unique identifier for this worker process.
        manager (Manager): The Manager instance for communication and shared memory access.
    """
    worker_mcts = create_multiprocessed_mcts(worker_id, manager)

    state = OthelloGame().get_init_board()

    to_play = -1  # Example: Player -1 starts
    start_time = time.time()

    for i in tqdm(range(100), desc=f"Worker {worker_id}"):
        worker_mcts.run_search(state, to_play)

    elapsed_time = time.time() - start_time
    print(f"Worker {worker_id}: {i+1} runs completed in {elapsed_time:.4f} sec")
    print(f"Worker {worker_id}: Avg time per turn = {(elapsed_time)/(i+1):.3f} sec")

    return worker_id




def main():
    """
    Main function to start the manager and worker processes.

    This function initializes the neural network model, creates the Manager and worker processes,
    and coordinates their execution.
    """
    h = Hyperparameters()
    g = OthelloGame()
    s = g.get_init_board()
    current_player = -1


    model = OthelloZeroModel(g.rows, g.get_action_size(), h.Neural_Network["device"])
    

    manager = init_manager(model, h)
    manager_process = init_manager_process(manager)

    # Start Worker Processes
    workers = []
    for i in range(4):
        worker_process = mp.Process(target=worker_process_function, args=(i, manager))
        workers.append(worker_process)
        worker_process.start()

    # Wait for Workers
    for worker in workers:
        worker.join()

    terminate_manager_process(manager_process)




if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()