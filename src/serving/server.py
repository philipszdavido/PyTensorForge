import asyncio
import logging
import os
import signal
import threading

from src.serving.app import APIApp
from src.serving.http import HTTPServer
from src.serving.model_server import ModelServer
from src.serving.security import is_loopback

log = logging.getLogger("ptf.server")

DEFAULT_UI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                              "web", "dist")


class InsecureConfiguration(ValueError):
    pass


def check_security(config, keys):
    if keys.enabled:
        return

    if config.security.allow_unauthenticated:
        if not is_loopback(config.host):
            log.warning("serving WITHOUT authentication on non-loopback address %s", config.host)
        return

    if not is_loopback(config.host):
        raise InsecureConfiguration(
            f"refusing to listen on {config.host} without API keys; configure security.api_keys / "
            f"api_keys_file / ${config.security.api_keys_env}, or set security.allow_unauthenticated"
        )

    config.security.allow_unauthenticated = True
    keys.allow_unauthenticated = True
    log.warning("no API keys configured; accepting unauthenticated requests on loopback %s", config.host)


class APIServer:

    def __init__(self, config):
        self.config = config.validate()
        self.models = ModelServer(self.config)
        ui_dir = self.config.ui.directory or DEFAULT_UI_DIR
        self.app = APIApp(self.models, self.config, ui_directory=ui_dir)
        check_security(self.config, self.app.keys)

        cors = {"Server": "pytensorforge"}
        self.http = HTTPServer(self.app, self.config.host, self.config.port, self.config.limits,
                               on_request_done=self.app.on_request_done, extra_headers=cors)
        self.http.on_connection_change = lambda n: self.app.metrics.open_connections.set(n)
        self._loop = None
        self._thread = None
        self._stopped = None
        self._stop_requested = None

    @property
    def port(self):
        return self.http.bound_port

    async def _run(self, ready=None, install_signals=True):
        self._loop = asyncio.get_running_loop()
        self._stop_requested = asyncio.Event()

        await self._loop.run_in_executor(None, self.models.preload)
        await self.http.start()
        log.info("listening on http://%s:%d", self.config.host, self.port)

        if install_signals:
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    self._loop.add_signal_handler(sig, self._stop_requested.set)
                except (NotImplementedError, RuntimeError):
                    pass

        if ready is not None:
            ready.set()

        await self._stop_requested.wait()
        log.info("shutting down: draining in-flight requests (up to %.0fs)", self.config.limits.shutdown_drain_s)

        await self.http.shutdown(self.config.limits.shutdown_drain_s)
        await self._loop.run_in_executor(None, self.models.shutdown, 5.0)
        self.app.close()
        log.info("shutdown complete")

    def serve_forever(self):
        asyncio.run(self._run())

    def start_background(self, timeout=60.0):
        ready = threading.Event()
        failure = []

        def target():
            try:
                asyncio.run(self._run(ready=ready, install_signals=False))
            except BaseException as exc:
                failure.append(exc)
                ready.set()

        self._thread = threading.Thread(target=target, name="ptf-api", daemon=True)
        self._thread.start()

        if not ready.wait(timeout):
            raise TimeoutError("server did not start in time")

        if failure:
            raise failure[0]

        return self

    def stop(self, timeout=30.0):
        if self._loop is not None and self._stop_requested is not None:
            self._loop.call_soon_threadsafe(self._stop_requested.set)

        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
