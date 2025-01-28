import numpy as np
import time
from src.neural_net.model import OthelloZeroModel
from src.othello.othello_game import OthelloGame
from src.config.hyperparameters import Hyperparameters
from src.mcts.worker import Worker
import multiprocessing as mp
import cProfile

from multiprocessing.shared_memory import SharedMemory
import numpy as np
import multiprocessing as mp
import time

class Manager:
    def __init__(self, model, state_shape):
        self.model = model
        self.state_shape = state_shape
        # Shared Memory Block für alle Spielzustände (maximal 23 Worker)
        self.shared_memory = SharedMemory(create=True, size=np.prod(state_shape) * 23 * np.float64().nbytes)
        self.shared_array = np.ndarray((23, *state_shape), dtype=np.float64, buffer=self.shared_memory.buf)

    def manage_workers(self, request_queue: mp.Queue, response_queues, batch_size=23, timeout=0.001):
        while True:
            batch_indices, worker_ids = [], []
            start_time = time.time()

            # Batching
            while len(batch_indices) < batch_size:
                try:
                    worker_id, state_index = request_queue.get(block=False)
                    batch_indices.append(state_index)  # Nur den Index merken
                    worker_ids.append(worker_id)
                except mp.queues.Empty:
                    if len(batch_indices) > 0 or (time.time() - start_time) >= timeout:
                        break

            # Model Prediction für Batch
            if batch_indices:
                batch_states = self.shared_array[batch_indices]  # Hole die Zustände aus Shared Memory
                policies, values = self.model.predict_batch(batch_states)
                
                # Ergebnisse zurück an die Worker senden
                for worker_id, policy, value in zip(worker_ids, policies, values):
                    response_queues[worker_id].put(
                        {"worker_id": worker_id, "policy": policy, "value": value}
                    )
    
    def cleanup(self):
        self.shared_memory.close()
        self.shared_memory.unlink()


def worker_process_function(worker_id, request_queue, response_queue, shared_memory_name, state_shape):
    """
    Funktion, die von jedem Worker-Prozess ausgeführt wird.

    Args:
        worker_id (int): Die eindeutige ID des Workers.
        request_queue (mp.Queue): Die Queue, um Anfragen an den Manager zu senden.
        response_queue (mp.Queue): Die Queue, um Antworten vom Manager zu empfangen.
        shared_memory_name (str): Name des Shared Memory Blocks.
        state_shape (tuple): Form des Spielzustands (z. B. (8, 8) für Othello).
    """
    from multiprocessing.shared_memory import SharedMemory
    import numpy as np
    from src.mcts.worker import Worker

    # Zugriff auf den Shared Memory Block
    shm = SharedMemory(name=shared_memory_name)
    shared_array = np.ndarray((23, *state_shape), dtype=np.float64, buffer=shm.buf)

    # Initialisiere den MCTS-Worker
    worker_mcts = Worker(worker_id=worker_id, request_queue=request_queue, response_queue=response_queue, shared_array=shared_array)

    # Initialen Spielzustand abrufen und zuweisen
    state = np.copy(shared_array[worker_id])  # Jeder Worker hat seinen Index im Shared Memory
    to_play = -1  # Beispiel: Spieler -1 beginnt

    start = time.time()
    # MCTS-Algorithmus ausführen
    for i in range(1):
        worker_mcts.run(state, to_play)
    print(f"time needed for processing = {time.time() - start} seconds")
        

    # Cleanup (optional)
    shm.close()




def main():
    # Initialisiere das Spiel und das Modell
    device = Hyperparameters.Neural_Network["device"]
    game = OthelloGame()
    model = OthelloZeroModel(game.rows, game.get_action_size(), device)
    model.eval()

    state_shape = (game.rows, game.columns)  # Form des Spielfelds

    # Manager und Shared Memory initialisieren
    manager = Manager(model=model, state_shape=state_shape)
    mp.set_start_method("spawn")

    num_workers = 20
    print(f"Num Workers are calculated by int(num_cores*0.85) => {num_workers}")

    request_queue = mp.Queue()
    response_queues = [mp.Queue() for _ in range(num_workers)]

    # Manager Prozess starten
    manager_process = mp.Process(
        target=manager.manage_workers,
        args=(request_queue, response_queues)
    )
    manager_process.start()

    # Worker Prozesse starten
    workers = []
    for i in range(num_workers):
        state = np.copy(game.get_init_board())
        worker_process = mp.Process(
            target=worker_process_function,
            args=(i, request_queue, response_queues[i], manager.shared_memory.name, state_shape),
        )
        workers.append(worker_process)
        worker_process.start()

    # Warten, bis die Worker fertig sind
    
    for worker_process in workers:
        worker_process.join()

    

    # Manager Prozess stoppen
    manager_process.terminate()

    # Shared Memory aufräumen
    manager.cleanup()





if __name__ == "__main__":

    #cProfile.run("main()", sort="time")
    main()
