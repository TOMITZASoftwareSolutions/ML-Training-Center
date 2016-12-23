import time
from rx import Observable, Observer
from rx.concurrency import NewThreadScheduler, ThreadPoolScheduler
import thread
import threading


class TrainingManager(Observable):
    def __init__(self):
        self.model = None

    def train(self, model, epochs=1):
        self.model = model
        return self.subscribe_on(ThreadPoolScheduler(max_workers=1))

    def get_current(self):
        return self.model

    def subscribe(self, subscriber):
        time.sleep(2)
        subscriber.on_next(self.model)
        self.model = None
        subscriber.on_completed()
