import csv
import os
import queue
import threading
import time

import h5py
import numpy as np


DEFAULT_COLUMNS = [
    ("time", "time_stamp"),
    ("set_T", "set_temperature"),
    ("T", "measured_temperature"),
    ("h_f", "heat_flux"),
    ("V", "voltage"),
    ("I", "current"),
    ("C_V", "calculated_voltage"),
]


class ExperimentDataSaver:
    """
    Persist experiment data in the background so instrument control never waits on disk IO.
    """

    def __init__(
        self,
        experiment_dir,
        r_vs_t,
        columns=None,
        flush_interval_s=5.0,
        batch_size=10,
    ):
        self.experiment_dir = experiment_dir
        self.r_vs_t = np.array(r_vs_t, dtype=float)
        self.columns = list(columns or DEFAULT_COLUMNS)
        self.flush_interval_s = max(float(flush_interval_s), 0.5)
        self.batch_size = max(int(batch_size), 1)

        self.csv_path = os.path.join(self.experiment_dir, "data.csv")
        self.h5_path = os.path.join(self.experiment_dir, "data.h5")
        self.r_vs_t_path = os.path.join(self.experiment_dir, "r_vs_t.csv")

        self._queue = queue.Queue()
        self._stop_token = object()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="experiment-data-saver")
        self._closed = False
        self._error = None
        self.rows_written = 0

    @property
    def error(self):
        return self._error

    def start(self):
        os.makedirs(self.experiment_dir, exist_ok=True)
        self._write_r_vs_t_snapshot()
        self._thread.start()
        self._ready.wait(timeout=10.0)
        self.raise_if_error()
        return self

    def enqueue(self, row):
        self.raise_if_error()
        if self._closed:
            raise RuntimeError("Experiment data saver is already closed.")
        self._queue.put(tuple(float(value) for value in row))

    def finalize(self, timeout=15.0):
        if self._closed:
            self.raise_if_error()
            return
        self._closed = True
        self._queue.put(self._stop_token)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise RuntimeError("Timed out while waiting for experiment data to finish saving.")
        self.raise_if_error()

    def raise_if_error(self):
        if self._error is not None:
            raise RuntimeError(f"Background data saver failed: {self._error}")

    def _write_r_vs_t_snapshot(self):
        with open(self.r_vs_t_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["resistivity", "temperature"])
            for resistivity, temperature in self.r_vs_t.T:
                writer.writerow([float(resistivity), float(temperature)])

    def _create_h5_datasets(self, h5_file):
        datasets = {}
        for _, dataset_name in self.columns:
            datasets[dataset_name] = h5_file.create_dataset(
                dataset_name,
                shape=(0,),
                maxshape=(None,),
                dtype="f8",
                chunks=True,
            )
        return datasets

    def _append_batch(self, batch, csv_writer, csv_file, datasets, h5_file):
        if not batch:
            return

        for row in batch:
            csv_writer.writerow(row)
        csv_file.flush()

        array_batch = np.asarray(batch, dtype=float)
        start_index = self.rows_written
        end_index = start_index + len(array_batch)
        for column_index, (_, dataset_name) in enumerate(self.columns):
            dataset = datasets[dataset_name]
            dataset.resize((end_index,))
            dataset[start_index:end_index] = array_batch[:, column_index]
        h5_file.flush()
        self.rows_written = end_index

    def _worker(self):
        batch = []
        last_flush = time.time()
        try:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as csv_file, h5py.File(
                self.h5_path, "w"
            ) as h5_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([column_name for column_name, _ in self.columns])
                datasets = self._create_h5_datasets(h5_file)
                self._ready.set()

                while True:
                    timeout = max(0.1, self.flush_interval_s - (time.time() - last_flush))
                    try:
                        item = self._queue.get(timeout=timeout)
                    except queue.Empty:
                        item = None

                    if item is self._stop_token:
                        self._append_batch(batch, csv_writer, csv_file, datasets, h5_file)
                        break

                    if item is not None:
                        batch.append(item)

                    if batch and (
                        len(batch) >= self.batch_size
                        or time.time() - last_flush >= self.flush_interval_s
                    ):
                        self._append_batch(batch, csv_writer, csv_file, datasets, h5_file)
                        batch = []
                        last_flush = time.time()

        except Exception as exc:
            self._error = exc
            self._ready.set()
