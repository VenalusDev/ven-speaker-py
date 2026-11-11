"""Small persistent LRU cache for synthesized audio chunks."""

import hashlib
import json
import os
import pathlib
import tempfile
import threading
from collections import OrderedDict


class AudioCache:
    """A disk cache with an LRU eviction queue and a configurable byte limit."""

    def __init__(self, directory: str | pathlib.Path, max_size_mb: float = 512) -> None:
        if max_size_mb < 0:
            raise ValueError("cache_size_mb must be non-negative")
        self.directory = pathlib.Path(directory).expanduser()
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self._lock = threading.RLock()
        self._queue: OrderedDict[str, pathlib.Path] = OrderedDict()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._load_queue()

    @staticmethod
    def make_key(*parts: object) -> str:
        encoded = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _load_queue(self) -> None:
        with self._lock:
            files = sorted(
                self.directory.glob("*.npy"),
                key=lambda path: (path.stat().st_mtime_ns, path.name),
            )
            for path in files:
                self._queue[path.stem] = path
            self._evict()

    def get(self, key: str):
        import numpy as np

        with self._lock:
            path = self._queue.get(key)
            if path is None or not path.is_file():
                self._queue.pop(key, None)
                return None
            try:
                audio = np.load(path, allow_pickle=False)
            except (OSError, ValueError):
                self._queue.pop(key, None)
                path.unlink(missing_ok=True)
                return None
            self._queue.move_to_end(key)
            try:
                os.utime(path, None)
            except OSError:
                pass
            return audio

    def put(self, key: str, audio) -> None:
        import numpy as np

        if self.max_size_bytes == 0:
            return
        with self._lock:
            path = self.directory / f"{key}.npy"
            fd, temporary_name = tempfile.mkstemp(prefix=".audio-", suffix=".npy", dir=self.directory)
            try:
                with os.fdopen(fd, "wb") as stream:
                    np.save(stream, np.asarray(audio, dtype=np.float32), allow_pickle=False)
                os.replace(temporary_name, path)
            finally:
                pathlib.Path(temporary_name).unlink(missing_ok=True)
            self._queue[key] = path
            self._queue.move_to_end(key)
            self._evict()

    def _evict(self) -> None:
        total = sum(path.stat().st_size for path in self._queue.values() if path.exists())
        while self._queue and total > self.max_size_bytes:
            _, oldest = self._queue.popitem(last=False)
            try:
                size = oldest.stat().st_size
                oldest.unlink(missing_ok=True)
                total -= size
            except OSError:
                pass
