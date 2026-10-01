import queue
import threading

_STOP = object()


class PrefetchLoader:

    def __init__(self, iterable_factory, prefetch_size=4):
        self.iterable_factory = iterable_factory
        self.prefetch_size = max(1, prefetch_size)
        self._queue = None
        self._thread = None
        self._error = None
        self._halt = threading.Event()

    def _put(self, item):
        while not self._halt.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue

        return False

    def _worker(self):
        try:
            for item in self.iterable_factory():
                if not self._put(item):
                    return
        except Exception as exc:
            self._error = exc
        finally:
            self._put(_STOP)

    def __iter__(self):
        self._queue = queue.Queue(maxsize=self.prefetch_size)
        self._error = None
        self._halt.clear()

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        try:
            while True:
                item = self._queue.get()

                if item is _STOP:
                    if self._error is not None:
                        raise self._error
                    return

                yield item
        finally:
            self.stop()

    def stop(self):
        self._halt.set()

        if self._thread is not None and self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
