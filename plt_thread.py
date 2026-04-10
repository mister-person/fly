from multiprocessing import Process, Queue
import multiprocessing
from queue import Empty
from threading import Thread
import matplotlib.pyplot as plt

class ThreadedPlot:
    def __init__(self):
        mp_context = multiprocessing.get_context("spawn")
        self.queue = mp_context.Queue()
        process = Process(target=self.start, args = [self.queue])
        process.start()

    def start(self, queue: multiprocessing.Queue):
        plt.ion()
        plt.show()
        plt.figure(1)
        while True:
            try:
                for x in range(50):
                    msg_type, msg = queue.get_nowait()
                    if msg_type == "plot":
                        plt.plot(*msg[0], **msg[1])
                    if msg_type == "clf":
                        plt.clf(*msg[0], **msg[1])
                    if msg_type == "cla":
                        plt.cla(*msg[0], **msg[1])

            except Empty:
                pass

            plt.pause(1/20)

            if not plt.fignum_exists(1):
                break

    def plot(self, *args, **kwargs):
        self.queue.put(("plot", (args, kwargs)))

    def clf(self, *args, **kwargs):
        self.queue.put(("clf", (args, kwargs)))

    def cla(self, *args, **kwargs):
        self.queue.put(("cla", (args, kwargs)))
