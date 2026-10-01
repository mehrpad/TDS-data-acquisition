"""Keep console output visible and save each run's transcript asynchronously."""
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import platform
import queue
import sys
import threading


class _ConsoleTee:
    def __init__(self, original, capture, channel):
        self.original, self.capture, self.channel = original, capture, channel

    def write(self, text):
        result = self.original.write(text) if self.original is not None else len(text)
        self.capture.write(self.channel, text)
        return result

    def flush(self):
        if self.original is not None:
            self.original.flush()
        self.capture.flush_fragment(self.channel)

    def __getattr__(self, name):
        return getattr(self.original, name)


class _LogWriter:
    def __init__(self, path):
        # Fail before hardware starts if the results log cannot be opened.
        self.stream = path.open('a', encoding='utf-8', buffering=1)
        self.queue = queue.Queue()
        self.error = None
        self.thread = threading.Thread(target=self._write, name='experiment-log-writer', daemon=True)
        self.thread.start()

    def _write(self):
        try:
            while True:
                text = self.queue.get()
                if text is None:
                    break
                self.stream.write(text)
                self.stream.flush()
        except Exception as exc:
            self.error = exc
        finally:
            self.stream.close()

    def close(self):
        self.queue.put(None)
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError('Timed out flushing the experiment log.')
        if self.error is not None:
            raise RuntimeError(f'Experiment log writer failed: {self.error}')


class RunConsoleCapture:
    """Capture preparatory console messages and tee an active run to tds_log.txt.

    The bounded prelude contains messages since the previous run ended, including
    curve loading and T0 calibration. Previous experiments are never copied into
    the next run's file. Capture stdout and stderr without replacing their normal
    destinations, and restore both streams when the application exits.
    """
    def __init__(self, prelude_limit=1_000_000):
        self.limit = prelude_limit
        self.history = deque()
        self.history_size = 0
        self.truncated = False
        self.fragments = {}
        self.lock = threading.RLock()
        self.writer = None
        self.stdout = self.stderr = None

    def __enter__(self):
        self.stdout, self.stderr = sys.stdout, sys.stderr
        sys.stdout = _ConsoleTee(self.stdout, self, 'stdout')
        sys.stderr = _ConsoleTee(self.stderr, self, 'stderr')
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.end_run('Application exiting')
        finally:
            if isinstance(sys.stdout, _ConsoleTee) and sys.stdout.capture is self:
                sys.stdout = self.stdout
            if isinstance(sys.stderr, _ConsoleTee) and sys.stderr.capture is self:
                sys.stderr = self.stderr

    def _record(self, channel, text, timestamp=None, thread_name=None):
        timestamp = timestamp or datetime.now().astimezone().isoformat(timespec='milliseconds')
        line = f'[{timestamp}] [{channel}] [{thread_name or threading.current_thread().name}] {text}\n'
        if self.writer is not None:
            self.writer.queue.put_nowait(line)
        else:
            self.history.append(line)
            self.history_size += len(line)
            while self.history and self.history_size > self.limit:
                self.history_size -= len(self.history.popleft())
                self.truncated = True

    def write(self, channel, text):
        with self.lock:
            key = (threading.get_ident(), channel)
            timestamp, name, pending = self.fragments.get(key, (
                datetime.now().astimezone().isoformat(timespec='milliseconds'),
                threading.current_thread().name, ''))
            parts = (pending + text).split('\n')
            for line in parts[:-1]:
                self._record(channel, line, timestamp, name)
            if parts[-1]:
                self.fragments[key] = (timestamp, name, parts[-1][-self.limit:])
            else:
                self.fragments.pop(key, None)

    def flush_fragment(self, channel):
        with self.lock:
            pending = self.fragments.pop((threading.get_ident(), channel), None)
            if pending:
                timestamp, name, text = pending
                self._record(channel, text, timestamp, name)

    def _flush_fragments(self):
        for (_, channel), (timestamp, name, text) in self.fragments.items():
            self._record(channel, text, timestamp, name)
        self.fragments.clear()

    def begin_run(self, directory, metadata):
        with self.lock:
            if self.writer is not None:
                raise RuntimeError('An experiment log is already active.')
            self._flush_fragments()
            path = Path(directory) / 'tds_log.txt'
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = _LogWriter(path)
            header = {'results_directory': str(path.parent.resolve()),
                      'python': sys.version, 'platform': platform.platform(),
                      'executable': sys.executable, 'run_metadata': metadata}
            writer.queue.put('=== TDS run log ===\n' + json.dumps(header, indent=2, default=str) + '\n')
            writer.queue.put('=== Preparation / calibration console messages ===\n')
            if self.truncated:
                writer.queue.put('[Earlier preparation output omitted: bounded history exceeded.]\n')
            for line in self.history:
                writer.queue.put(line)
            writer.queue.put('=== Experiment console messages ===\n')
            self.history.clear()
            self.history_size = 0
            self.truncated = False
            self.writer = writer
            self._record('event', 'Experiment log opened')
            return path

    def end_run(self, reason):
        with self.lock:
            if self.writer is None:
                return
            self._flush_fragments()
            self._record('event', reason)
            writer, self.writer = self.writer, None
        # Disk waits happen only after the experiment has stopped, outside the
        # capture lock, so other console writers cannot block instrument control.
        writer.close()
