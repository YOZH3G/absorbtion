"""Cooperative calculation tasks; this module never calls Tk or Matplotlib."""

from queue import Empty, Queue
from threading import Event, Thread


class CalculationCancelled(Exception):
    pass


def check_cancelled(cancel):
    if cancel is not None and cancel.is_set():
        raise CalculationCancelled()


class BackgroundTask:
    def __init__(self, calculate):
        self.cancel = Event()
        self.progress = 0.0
        self.results = Queue()
        self.thread = Thread(target=self._run, args=(calculate,), daemon=True)
        self.thread.start()

    def _run(self, calculate):
        try:
            value = calculate(self.cancel, self._progress)
            check_cancelled(self.cancel)
            self.results.put((value, None))
        except Exception as error:
            self.results.put((None, error))

    def _progress(self, fraction):
        self.progress = max(0.0, min(1.0, fraction))

    def poll(self):
        try:
            return self.results.get_nowait()
        except Empty:
            return None
