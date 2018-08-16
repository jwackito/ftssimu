import numpy as np
import simpy


class Transfer():
    def __init__(self, size, submitted_at, env):
        self.size = size
        self.submitted_at = submitted_at
        self.started_at = None
        self.transferred_at = None
        self.rate_history = []
        self.elapsed_time = 0
        self.bytes_remaining = size

    def step(self, env, link_rate):
        self.bytes_remaining -= link_rate
        self.rate_history.append(link_rate)
        if self.bytes_remaining < 0:
            # transfer is done
            self.elapsed = env.now - self.started_at
                


class Link():
    def __init__(self, bandwidth, limit, env):
        self.bandwidth = bandwidth
        self.limit = limit
        self.link = []
        self.env = env




class FTS():
    def __init__(self):
        queue = []
        active = 0
    
    def submit(self, transfers):
        for t in transfers:
            queue.append(t)

    def queue_empty(self):
        return len(self.queue) == 0


if __name__ == "__main__":
    # execute only if run as a script
    fts = FTS()
    link = Link((250*2**20)//8, 100)
    t = Transfer(1*2**30, 1)
    fts.submit([t])
    
