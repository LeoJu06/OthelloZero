import time
from src.mcts.mcts import MCTS
import multiprocessing as mp
from src.mcts.node import Node
from src.config.hyperparameters import Hyperparameters



class Worker:
    def __init__(self, worker_id, root_node=None):
        self.worker_id = worker_id
        self.moves_played = 0
        self.inputs = []
        self.outputs = []
        if root_node is None:
            self.mcts = MCTS(Node(prior=float("inf")))
        else:
            self.mcts = MCTS(root_node=root_node)

    def run_mcts_search(self, request_queue, response_queue, num_simulations=Hyperparameters.MCTS["num_simulations"]):
        for _ in range(num_simulations):

            #print(f"Worker={self.worker_id}, simulation={_}")
            node = self.mcts.root
            search_path = [node]

            # Selection phase
            while node.is_expanded():
                action, node = node.select_child()
                search_path.append(node)

            # Expansion and evaluation
            if node.is_terminal_state():
                value = node.board.determine_winner()
            else:
                request_queue.put((self.worker_id, search_path[-1].board.board))
                while True:
                    try:
                        response = response_queue.get_nowait()
                        if response["worker_id"] == self.worker_id:  # Check worker ID
                            policy, value = response["policy"], response["value"]
                            node.expand(policy)
                            break
                    except mp.queues.Empty:
                        time.sleep(0.01)  # Avoid busy-waiting

            # Backpropagation phase
            for node in search_path:
                node.value += value
                node.visits += 1

        print(f"Woker={self.worker_id} finished")

    def perform_self_play(self, request_queue, response_queue):

        while not self.mcts.root.board.is_terminal_state():

            self.run_mcts_search(request_queue=request_queue, response_queue=response_queue)
            
            best_move, child = self.mcts.get_best_move()
            pass


