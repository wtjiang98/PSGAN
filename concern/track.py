import time

import torch


class Track:
    def __init__(self):
        self.log_point = time.time()
        self.enable_track = False

    def track(self, mark):
        if not self.enable_track:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            print("{} memory:".format(mark), torch.cuda.memory_allocated() / 1024 / 1024, "M")
        print("{} time cost:".format(mark), time.time() - self.log_point)
        self.log_point = time.time()
