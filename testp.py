import multiprocessing
from multiprocessing import Manager
import time

class MyClass:
    def __init__(self, qs):
        self.queues = qs

    def worker(self, wn):
        # Implement your worker method here
        for i in range(5):
            for j, queue in enumerate(self.queues):

                queue.put(f"Q{j}Worker {wn} - Iteration {i}")
                time.sleep(0.25)

class DataManager:

    def __init__(self, num_envs):

        self.queues = []
        for _ in range(2):
            self.queues.append(multiprocessing.Queue())
        loader = MyClass(self.queues)

        # Create 4 processes
        workers = []
        for i in range(num_envs):
            wp = multiprocessing.Process(
                target=loader.worker,
                args=(i,)
            )
            workers.append(wp)
            wp.start()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    dm = DataManager(4)

    while True:
        for queue in dm.queues:
            print(queue.get())


