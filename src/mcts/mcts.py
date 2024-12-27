import multiprocessing as mp
import torch
import torch.nn as nn
import numpy as np
import time
import src.utils.logger_config as lg
from src.config.hyperparameters import Hyperparameters
from src.mcts.node import Node
from src.neural_net.model import neural_network_evaluate, NeuralNetwork


class MCTS:
    """
    Monte Carlo Tree Search (MCTS) implementation with multiprocessing capabilities.
    """

    def __init__(self, worker_id=None, root_node=None):
        """
        Initialize the MCTS object.

        Args:
            worker_id (int, optional): Unique identifier for the worker.
            root_node (Node, optional): The root node of the MCTS tree. If not provided, a new root node is created.
        """
        self.worker_id = worker_id  # Unique ID for the worker, used to track its tasks and responses.
        if root_node is None:
            self.root = Node(
                prior=float("inf")
            )  # Initialize the root node with infinite prior.
        else:
            self.root = root_node  # Use the provided root node.
        pass  # Explicitly indicating that no other initialization is required.

    def _search_multicored(self, request_queue, response_queue):
        """
        Perform MCTS search as part of a worker process.

        Args:
            request_queue (mp.Queue): Queue for sending board states to the manager.
            response_queue (mp.Queue): Queue for receiving policy and value predictions from the manager.
        """
        num_simulations = Hyperparameters.MCTS["num_simulations"]
        for simulation in range(num_simulations):
            #lg.logger_mcts.debug(
                #"Worker=%s, simulation=> %s/%s",
                #self.worker_id,
                #simulation,
                #num_simulations,
            #)

            # Start the search at the root node.
            node = self.root
            search_path = [node]  # Track the path of nodes visited during the search.

            # Selection phase: Traverse the tree to find a leaf node.
            while node.is_expanded():
                (
                    action,
                    node,
                ) = (
                    node.select_child()
                )  # Select the child node with the best UCB score.
                search_path.append(node)  # Add the selected node to the search path.

            # Evaluate the terminal state or expand the leaf node.
            value = None
            if node.is_terminal_state():  # Check if the game is over at this node.
                value = (
                    node.board.determine_winner()
                )  # Determine the winner of the game.
            if (
                value is None
            ):  # If not terminal, request neural network predictions for the leaf node.
                leaf_node = search_path[-1]  # The current leaf node.
                request_queue.put(
                    (self.worker_id, leaf_node.board.board)
                )  # Send the board state to the manager.

                # Wait for the manager's response.
                while True:
                    try:
                        response = (
                            response_queue.get_nowait()
                        )  # Non-blocking get from the response queue.
                        if (
                            response["worker_id"] == self.worker_id
                        ):  # Ensure the response matches this worker.
                            policy, value = response["policy"], response["value"]

                            # lg.logger_mcts.debug("Worker=%s: state:%s", self.worker_id, search_path[-1].board.print_board(to_console=False))
                            # lg.logger_mcts.debug("Policy => %s", policy)
                            node.expand(
                                policy
                            )  # Expand the node with the predicted policy.
                            break
                    except mp.queues.Empty:
                        time.sleep(0.01)  # Avoid busy-waiting by sleeping briefly.

            # Backpropagation phase: Update the value and visit count along the search path.
            for node in search_path:
                node.value += value
                node.visits += 1

    def _manage_multicored_search(self, request_queue, response_queues, batch_size=10):
        """
        Manage batching of neural network predictions for multiple workers.

        Args:
            request_queue (mp.Queue): Queue for receiving board states from workers.
            response_queues (list[mp.Queue]): List of queues for sending responses back to workers.
            batch_size (int): Maximum number of board states to batch together for a single prediction.
        """
        # Initialize the neural network model on the appropriate device.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = NeuralNetwork().to(device)
        model.eval()  # Set the model to evaluation mode.

        while True:
            batch = []  # Collect board states for a batch prediction.
            worker_ids = []  # Track which workers submitted each board state.

            # Collect requests until the batch size is reached or no more requests are available.
            while len(batch) < batch_size and not request_queue.empty():
                (
                    worker_id,
                    state,
                ) = request_queue.get()  # Retrieve the worker ID and board state.
                batch.append(state)  # Add the board state to the batch.
                worker_ids.append(worker_id)  # Track the corresponding worker ID.

            if batch:  # If there are states to process:
                policies, values = neural_network_evaluate(
                    batch, model, device
                )  # Predict policies and values.

                # Send the results back to the respective workers.
                for worker_id, policy, value in zip(worker_ids, policies, values):
                    response_queues[worker_id].put(
                        {"worker_id": worker_id, "policy": policy, "value": value}
                    )

    def run_multicored(self):
        """
        Run the multi-core MCTS simulation by spawning manager and worker processes.
        """
        mp.set_start_method(
            "spawn"
        )  # Use 'spawn' to start new processes (recommended for compatibility).

        # Determine the number of workers based on available CPU cores.
        num_workers = int(mp.cpu_count() * 0.85)
        print(f"Num Workers are calculated by int(num_cores*0.85) => {num_workers}")

        # Create queues for inter-process communication.
        request_queue = (
            mp.Queue()
        )  # Shared request queue for sending tasks to the manager.
        response_queues = [
            mp.Queue() for _ in range(num_workers)
        ]  # Separate response queues for each worker.

        # Start the manager process.
        manager = mp.Process(
            target=self._manage_multicored_search, args=(request_queue, response_queues)
        )
        manager.start()

        # Start worker processes.
        workers = []
        for i in range(num_workers):
            worker_mcts = MCTS(
                worker_id=i
            )  # Create a new MCTS instance for each worker.
            worker = mp.Process(
                target=worker_mcts._search_multicored,
                args=(request_queue, response_queues[i]),
            )
            workers.append(worker)
            worker.start()

        # Wait for all worker processes to finish.
        for worker in workers:
            worker.join()

        # Terminate the manager process.
        manager.terminate()


if __name__ == "__main__":
    # Create an instance of MCTS and run the multi-core simulation.
    mcts = MCTS()
    mcts.run_multicored()
