import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PyQt6 import QtWidgets

from tds_control import app, config_io, material_profiles, tds_experiment as ctl
from tools.update_nicr50_run130 import derive_partial_table


PROFILE = Path(__file__).resolve().parents[1]/'files/material_profiles/NiCr_50.json'


def fixture():
    t = np.r_[np.full(160,23.),np.repeat([130.,150.,170.,190.,200.],30)]
    applied = .001+.0004*t
    data = pd.DataFrame(dict(time=np.arange(len(t))*2.,T=t,set_T=t+5.,
        V=applied*15.,I=applied/6.,P=(applied/6.)**2*90.))
    diag = pd.DataFrame(dict(time=data.time+.001,status='accepted',raw_temperature_c=t,
        applied_current_a=applied,accepted_current_a=applied+.03,range_change_count=0))
    return data,diag


class NiCr50Run130Tests(unittest.TestCase):
    def test_partial_table_pairs_applied_commands_and_rejects_noise_and_range_changes(self):
        data,diag = fixture()
        diag.loc[160,'status']='invalid_backoff'
        diag.loc[190:,'range_change_count']=1
        data.loc[220,'T']=400.
        expected,_ = derive_partial_table(data,diag)
        for i in [160,190,191,192,220]:
            diag.loc[i,'applied_current_a']=2.
        actual,_ = derive_partial_table(data,diag)
        self.assertEqual(expected,actual)
        self.assertEqual(actual[-1]['temperature_c'],200.)
        self.assertAlmostEqual(actual[-1]['current_a'],.081)
        with self.assertRaisesRegex(ValueError,'align'):
            derive_partial_table(data.iloc[:-1],diag)
        with self.assertRaisesRegex(ValueError,'coverage'):
            derive_partial_table(data.iloc[:220],diag.iloc[:220])

    def test_requested_limits_and_actual_user_pi_survive_config_build(self):
        p = json.loads(PROFILE.read_text())
        c = ctl.build_control_config(p)
        self.assertEqual((c['max_current'],c['max_wire_current_a'],c['max_power_w']), (3.,3.,15.))
        self.assertEqual((c['compliance_voltage'],c['max_sample_voltage']), (5.,5.))
        self.assertEqual(c['pid_gain_schedule'],[])
        self.assertEqual(ctl.pid_gains_for_current(c,.001),(.001,.001/600,0.))
        self.assertEqual(c['pid_integral_time_s'],600.)
        # The former hidden 30 mA cutoff must not defeat the selected settings.
        ctl._enforce_electrical_safety(4.,.04,c)
        for v,i in [(5.01,.04),(4.,3.01),(5.,3.)]:
            with self.assertRaises(ctl.ExperimentSafetyError):
                ctl._enforce_electrical_safety(v,i,c)
        with self.assertRaises(ctl.ExperimentSafetyError):
            ctl._enforce_temperature_safety(600.1,c)

    def test_new_coverage_and_extension_are_not_mislabeled_as_600_measurement(self):
        p = json.loads(PROFILE.read_text())
        info = p['current_feedforward_provenance']['run130_revision']
        self.assertLess(info['measured_indicated_range_c'][1],205.)
        self.assertGreater(info['high_temperature_command_offset_a'],.015)
        self.assertFalse(info['independent_temperature_calibration'])
        table = p['current_feedforward_table']
        self.assertTrue(all(not x['measured_on_this_wire'] for x in table
                            if x['temperature_c']>info['measured_indicated_range_c'][1]))
        self.assertTrue(all(b['current_a']>a['current_a'] for a,b in zip(table,table[1:])))
        self.assertGreater(ctl.current_feedforward_for_temperature(p,200.),.075)


class NiCr50Run130GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_loading_profile_replaces_old_limits_and_shows_voltage_and_flat_gains(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath('counter.txt').write_text('1')
            with patch.object(config_io,'CONFIG_PATH',root/'config.toml'), \
                 patch.object(app,'EXPERIMENT_COUNTER_PATH',root/'counter.txt'), \
                 patch.object(app,'ensure_runtime_dirs'), \
                 patch.object(material_profiles,'PROFILES_DIR',PROFILE.parent):
                window = QtWidgets.QMainWindow()
                ui = app.Ui_TDS(ctl.build_control_config(dict(DMM_speed=10,experiment_frequency=.5)))
                ui.setupUi(window)
                try:
                    ui.material_profile_combo.setCurrentText('NiCr_50')
                    ui.load_material_profile()
                    self.assertEqual(float(ui.max_current.text()),3.)
                    self.assertEqual(float(ui.max_power.text()),15.)
                    self.assertEqual(ui.voltage_limits_label.text(),'Voltage limits: PSU 5 V; sample 5 V')
                    self.assertEqual(float(ui.pid_kp_edit.text()),.001)
                    self.assertEqual(float(ui.pid_ti_edit.text()),600.)
                    self.assertEqual(ui.config['pid_gain_schedule'],[])
                    saved = config_io.load_config()
                    self.assertEqual(saved['compliance_voltage'],5.)
                    self.assertEqual(saved['max_wire_current_a'],3.)
                finally:
                    ui.timer_error.stop()
                    window.close()


if __name__=='__main__':
    unittest.main()
