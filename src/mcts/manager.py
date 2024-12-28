import torch 
from src.neural_net.model import NeuralNetwork, neural_network_evaluate
from src.mcts.worker import Worker
import multiprocessing as mp

class Manager:
    def __init__(self, model):
        """
        Initialize the manager with batching and coordination logic.
        """
        self.model = model
        


    def manage_wworkers(self, request_queue, response_queue, batch_size=10):
        while True:
            batch, worker_ids = [], []
            while len(batch) < batch_size and not request_queue.empty():
                worker_id, state = request_queue.get()
                batch.append(state)
                worker_ids.append(worker_id)

            if batch:
                policies, values = neural_network_evaluate(batch, self.model)
                for worker_id, policy, value in zip(worker_ids, policies, values):
                    response_queue[worker_id].put({"worker_id": worker_id, "policy": policy, "value": value})


if __name__ == "__main__":
    import pynvml

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork().to(device)
    model.eval()

    print(f"Allocated: {torch.cuda.memory_allocated()} bytes")
    print(f"Cached: {torch.cuda.memory_reserved()} bytes")
    # Initialize NVML
    pynvml.nvmlInit()

    # Get the first GPU handle (adjust index for multiple GPUs)
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    # Get memory info
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)

    print(f"Total VRAM: {info.total / (1024**2):.2f} MB")
    print(f"Used VRAM: {info.used / (1024**2):.2f} MB")
    print(f"Free VRAM: {info.free / (1024**2):.2f} MB")


        
    
    manager = Manager(model=model)
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
    manager_process = mp.Process(
        target=manager.manage_wworkers, args=(request_queue, response_queues)
    )
    manager_process.start()

    # Start worker processes.
    workers = []
    for i in range(num_workers):
        worker_mcts = Worker(
            worker_id=i
        )  # Create a new MCTS instance for each worker.
        worker_process = mp.Process(
            target=worker_mcts.run_mcts_search,
            args=(request_queue, response_queues[i]),
        )
        workers.append(worker_process)
        worker_process.start()

    # Wait for all worker processes to finish.
    for worker_process in workers:
        worker_process.join()

    # Terminate the manager process.
    manager_process.terminate()

    print(f"Allocated: {torch.cuda.memory_allocated()} bytes")
    print(f"Cached: {torch.cuda.memory_reserved()} bytes")
