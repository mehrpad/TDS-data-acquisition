import contextlib
import io
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PyQt6 import QtWidgets

from tds_control.app import SignalEmitter, WorkerThread, Ui_TDS
from tds_control import app as gui, config_io, material_profiles
from tds_control.run_logging import RunConsoleCapture
from tds_control import tds_experiment as ctl


class RunLoggingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_captures_preparation_stdout_stderr_thread_output_and_restores_console(self):
        output, errors = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as folder, \
             contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            old_stdout, old_stderr = sys.stdout, sys.stderr
            with RunConsoleCapture() as capture:
                print('T0 search: trying 0.0010 A')
                print('T0 warning: noisy', file=sys.stderr)
                path = capture.begin_run(folder, {'profile_name': 'NiCr_50', 'startup_current': .001})
                worker = threading.Thread(target=lambda: print('Current command 0.001 A'), name='instrument-test')
                worker.start(); worker.join()
                print('partial', end='', flush=True)
                capture.end_run('Stopped by user')
            self.assertIs(sys.stdout, old_stdout)
            self.assertIs(sys.stderr, old_stderr)
            saved = path.read_text(encoding='utf-8')
            self.assertIn('T0 search: trying 0.0010 A', saved)
            self.assertIn('[stderr]', saved)
            self.assertIn('T0 warning: noisy', saved)
            self.assertIn('[instrument-test] Current command 0.001 A', saved)
            self.assertIn('partial', saved)
            self.assertIn('Stopped by user', saved)
            self.assertIn('"profile_name": "NiCr_50"', saved)
            self.assertIn('T0 search: trying 0.0010 A', output.getvalue())
            self.assertIn('T0 warning: noisy', errors.getvalue())

    def test_second_run_has_own_preparation_and_does_not_copy_previous_experiment(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            with RunConsoleCapture() as capture:
                first = capture.begin_run(Path(folder)/'1', {})
                print('FIRST experiment output')
                capture.end_run('First completed')
                print('SECOND T0 calibration')
                second = capture.begin_run(Path(folder)/'2', {})
                print('SECOND experiment output')
                capture.end_run('Second completed')
            saved = second.read_text()
            self.assertIn('SECOND T0 calibration', saved)
            self.assertIn('SECOND experiment output', saved)
            self.assertNotIn('FIRST experiment output', saved)
            self.assertNotIn('First completed', saved)
            self.assertNotIn('SECOND', first.read_text())

    def test_bounded_preparation_marks_truncation_but_active_log_is_complete(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            with RunConsoleCapture(prelude_limit=200) as capture:
                for n in range(10):
                    print(f'old preparation {n}')
                path = capture.begin_run(folder, {})
                for n in range(10):
                    print(f'active output {n}')
                capture.end_run('Completed')
            text = path.read_text()
            self.assertIn('bounded history exceeded', text)
            self.assertNotIn('old preparation 0', text)
            self.assertIn('old preparation 9', text)
            for n in range(10):
                self.assertIn(f'active output {n}', text)

    def test_worker_traceback_saved_after_data_finalization(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            with RunConsoleCapture() as capture:
                path = capture.begin_run(folder, {})
                def fail(emitter):
                    print('Power supply output switched OFF.')
                    print('Data saver finalized.')
                    raise ctl.ExperimentSafetyError('Measured temperature 701.25 C exceeded Maximum Temperature 600 C.')
                result = []
                worker = WorkerThread(fail, SignalEmitter())
                worker.finished.connect(result.append)
                worker.run()
                capture.end_run(f'Experiment stopped with error: {result[0]}')
            text = path.read_text()
            self.assertIn('Data saver finalized.', text)
            self.assertIn('Traceback (most recent call last):', text)
            self.assertIn('ExperimentSafetyError', text)
            self.assertIn('701.25', text)
            self.assertIsInstance(result[0], ctl.ExperimentSafetyError)

    def test_raw_trip_pair_is_reported_before_exception(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            config = ctl.build_control_config({'max_temperature_c': 600,
                '_last_acquisition': {'voltage': .190, 'current': .002},
                '_active_dmm_volt_range': 2., '_active_dmm_curr_range': .02})
            with RunConsoleCapture() as capture:
                path = capture.begin_run(folder, {})
                with self.assertRaises(ctl.ExperimentSafetyError):
                    ctl._temperature_from_resistance(95., lambda r: 701.25, config)
                capture.end_run('Error')
            text = path.read_text()
            self.assertIn('R=95.000000000 Ohm', text)
            self.assertIn("'voltage': 0.19", text)
            self.assertIn("'current': 0.002", text)
            self.assertIn('current range=0.02 A', text)

    def test_log_writer_failure_is_reported_and_streams_restored(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            original = sys.stdout
            with self.assertRaisesRegex(RuntimeError, 'disk failure'):
                with RunConsoleCapture() as capture:
                    capture.begin_run(folder, {})
                    capture.writer.error = OSError('disk failure')
            self.assertIs(sys.stdout, original)

    def test_conflicting_start_and_unwritable_log_do_not_replace_active_writer(self):
        with tempfile.TemporaryDirectory() as folder:
            capture = RunConsoleCapture()
            with patch.object(Path, 'open', side_effect=PermissionError('Cannot open log')):
                with self.assertRaises(PermissionError):
                    capture.begin_run(folder, {})
            self.assertIsNone(capture.writer)
            capture.begin_run(folder, {})
            original = capture.writer
            with self.assertRaisesRegex(RuntimeError, 'already active'):
                capture.begin_run(Path(folder)/'second', {})
            self.assertIs(capture.writer, original)
            capture.end_run('Completed')

    def test_gui_temperature_and_current_runs_open_log_and_close_after_final_message(self):
        for mode in ('TEMPERATURE', 'CURRENT'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder, \
                 contextlib.redirect_stdout(io.StringIO()), \
                 patch.object(gui, 'DATA_DIR', Path(folder)), \
                 patch.object(gui, 'EXPERIMENT_COUNTER_PATH', Path(folder)/'counter.txt'), \
                 patch.object(gui, 'ensure_runtime_dirs'), \
                 patch.object(config_io, 'CONFIG_PATH', Path(folder)/'config.toml'), \
                 patch.object(config_io, 'ensure_runtime_dirs'), \
                 patch.object(material_profiles, 'PROFILES_DIR', Path(folder)/'profiles'), \
                 patch.object(gui, 'WorkerThread') as worker:
                settings = ctl.build_control_config({'experiment_mode': mode,
                    'experiment_frequency': .5, 'DMM_speed': 10})
                with RunConsoleCapture() as capture:
                    window = QtWidgets.QMainWindow()
                    ui = Ui_TDS(settings, console_capture=capture)
                    ui.setupUi(window)
                    ui.r_vs_t = np.array([[88., 95.], [23., 600.]])
                    ui.t_zero_calibrated = True
                    try:
                        print('Calibration completed before Start')
                        ui.start_clicked()
                        worker.return_value.start.assert_called_once()
                        path = ui.current_experiment_dir/'tds_log.txt'
                        # The real background data saver finishes before the GUI
                        # receives the worker's terminal signal / traceback.
                        ui.data_saver.finalize()
                        print('Post-finalization console output')
                        ui.thread_finished(RuntimeError('example failure'))
                        saved = path.read_text(encoding='utf-8')
                        self.assertIn('Calibration completed before Start', saved)
                        self.assertIn('Post-finalization console output', saved)
                        self.assertIn('Experiment stopped with error: example failure', saved)
                        self.assertIn(f'"experiment_mode": "{mode}"', saved)
                        self.assertIsNone(capture.writer)
                    finally:
                        if ui.data_saver is not None:
                            ui.data_saver.finalize()
                        ui.timer_error.stop()
                        ui.update_timer.stop()
                        window.close()


if __name__ == '__main__':
    unittest.main()
