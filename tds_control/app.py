import os
import re
import sys
import traceback

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
import pyqtgraph as pg

from . import calibration
from . import config_io
from . import material_profiles
from . import tds_experiment
from .curve_io import load_resistance_temperature_file
from .data_saver import ExperimentDataSaver
from .paths import DATA_DIR, EXPERIMENT_COUNTER_PATH, ensure_runtime_dirs

RESISTIVITY_MODE_TOOLTIP = """How sample resistance is measured.

V_OVER_I: continuous R = V / I while the heating current flows.
OFFSET_CORRECTED: switches the supply off each cycle, reads the contact thermal EMF with no current flowing, and subtracts it from the sense voltage before dividing by the current.
FOUR_WIRE: switches the supply off each cycle and lets the voltage DMM source its own test current for a direct four-wire resistance reading. Requires the meter's source leads to be wired to the sample.

The two duty-cycled modes set the loop period from resistivity_heat_time_s + resistivity_measure_time_s, so the sample only heats for part of each cycle."""


class Ui_TDS(object):
    def __init__(self, data):
        self.target_temperature = 0
        self.index_plot = 0
        self.voltage = 0
        self.current = 0
        self.temperature = 0
        self.resistivity = 0
        self.commanded_current = 0
        self.sample_power = 0
        self.index_plot_start = 0
        self.plot_window_last60_selected = False
        self.plot_window_seconds = 60.0
        self.config = tds_experiment.build_control_config(data)
        self.experiment_params = []
        self.r_vs_t = None
        self.data_list = []
        self.worker_thread = None
        self.calibration_worker = None
        self._operation_running = False
        self.data_saver = None
        self.current_experiment_dir = None
        self.file_path = None
        self.experiment_name = 'TDS_test'
        self.t_zero_calibrated = False
        self.t0_calibration_warning = None
        self._active_experiment_mode = None
        self._program_text_by_mode = {
            "TEMPERATURE": (
                "{start_T=40;step_T=200;target_T=200;"
                "ramp_speed_min=10;hold_step_time_min=1}"
            ),
            "CURRENT": "{ramp_speed_min=0.01}",
        }
        ensure_runtime_dirs()
        if EXPERIMENT_COUNTER_PATH.exists():
            # Read the experiment counter
            with EXPERIMENT_COUNTER_PATH.open() as f:
                self.ex_counter = int(f.readlines()[0])
        else:
            # create a new txt file
            with EXPERIMENT_COUNTER_PATH.open('w') as f:
                f.write(str(1))  # Current time and date
                self.ex_counter = 1
        self.freq_acquisition = self.config['experiment_frequency']  # Frequency of data acquisition
        self.emitter = SignalEmitter()
        self.columns = [
            "time",  # Time in UNIX-readable format
            "set_T",  # Set temperature
            "T",  # Measured temperature
            "h_f",  # Heat flux
            "V",  # Voltage
            "I",  # Current
            "C_V",  # Calculated Power supply voltage
            "P",  # Measured sample power
            "R_ohm",  # Calculated Kelvin sample resistance
        ]

    def setupUi(self, TDS):
        TDS.setObjectName("TDS")
        TDS.resize(1020, 560)
        self.centralwidget = QtWidgets.QWidget(parent=TDS)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_183 = QtWidgets.QLabel(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_183.sizePolicy().hasHeightForWidth())
        self.label_183.setSizePolicy(sizePolicy)
        self.label_183.setObjectName("label_183")
        self.gridLayout.addWidget(self.label_183, 0, 0, 1, 1)
        self.ex_number = QtWidgets.QLineEdit(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ex_number.sizePolicy().hasHeightForWidth())
        self.ex_number.setSizePolicy(sizePolicy)
        self.ex_number.setMinimumSize(QtCore.QSize(0, 20))
        self.ex_number.setStyleSheet("QLineEdit{\n"
                                     "                                                background: rgb(223,223,233)\n"
                                     "                                                }\n"
                                     "                                            ")
        self.ex_number.setObjectName("ex_number")
        self.gridLayout.addWidget(self.ex_number, 0, 1, 1, 1)
        self.label_175 = QtWidgets.QLabel(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_175.sizePolicy().hasHeightForWidth())
        self.label_175.setSizePolicy(sizePolicy)
        self.label_175.setObjectName("label_175")
        self.gridLayout.addWidget(self.label_175, 1, 0, 1, 1)
        self.ex_name = QtWidgets.QLineEdit(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ex_name.sizePolicy().hasHeightForWidth())
        self.ex_name.setSizePolicy(sizePolicy)
        self.ex_name.setMinimumSize(QtCore.QSize(0, 20))
        self.ex_name.setMaximumSize(QtCore.QSize(16777215, 100))
        self.ex_name.setStyleSheet("QLineEdit{\n"
                                   "                                                background: rgb(223,223,233)\n"
                                   "                                                }\n"
                                   "                                            ")
        self.ex_name.setObjectName("ex_name")
        self.gridLayout.addWidget(self.ex_name, 1, 1, 1, 1)
        self.label_176 = QtWidgets.QLabel(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_176.sizePolicy().hasHeightForWidth())
        self.label_176.setSizePolicy(sizePolicy)
        self.label_176.setObjectName("label_176")
        self.gridLayout.addWidget(self.label_176, 2, 0, 1, 1)
        self.calib_temperature = QtWidgets.QLineEdit(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.calib_temperature.sizePolicy().hasHeightForWidth())
        self.calib_temperature.setSizePolicy(sizePolicy)
        self.calib_temperature.setMinimumSize(QtCore.QSize(100, 20))
        self.calib_temperature.setStyleSheet("QLineEdit{\n"
                                             "                                                background: rgb(223,223,233)\n"
                                             "                                                }\n"
                                             "                                            ")
        self.calib_temperature.setObjectName("calib_temperature")
        self.gridLayout.addWidget(self.calib_temperature, 2, 1, 1, 1)
        self.label_177 = QtWidgets.QLabel(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_177.sizePolicy().hasHeightForWidth())
        self.label_177.setSizePolicy(sizePolicy)
        self.label_177.setObjectName("label_177")
        self.gridLayout.addWidget(self.label_177, 3, 0, 1, 1)
        self.max_power = QtWidgets.QLineEdit(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.max_power.sizePolicy().hasHeightForWidth())
        self.max_power.setSizePolicy(sizePolicy)
        self.max_power.setMinimumSize(QtCore.QSize(100, 20))
        self.max_power.setStyleSheet("QLineEdit{\n"
                                       "                                                background: rgb(223,223,233)\n"
                                       "                                                }\n"
                                       "                                            ")
        self.max_power.setObjectName("max_power")
        self.gridLayout.addWidget(self.max_power, 3, 1, 1, 1)
        self.label_178 = QtWidgets.QLabel(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_178.sizePolicy().hasHeightForWidth())
        self.label_178.setSizePolicy(sizePolicy)
        self.label_178.setObjectName("label_178")
        self.gridLayout.addWidget(self.label_178, 4, 0, 1, 1)
        self.max_current = QtWidgets.QLineEdit(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.max_current.sizePolicy().hasHeightForWidth())
        self.max_current.setSizePolicy(sizePolicy)
        self.max_current.setMinimumSize(QtCore.QSize(100, 20))
        self.max_current.setStyleSheet("QLineEdit{\n"
                                       "                                                background: rgb(223,223,233)\n"
                                       "                                                }\n"
                                       "                                            ")
        self.max_current.setObjectName("max_current")
        self.gridLayout.addWidget(self.max_current, 4, 1, 1, 1)
        self.label_calibration_start_current = QtWidgets.QLabel(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_calibration_start_current.sizePolicy().hasHeightForWidth())
        self.label_calibration_start_current.setSizePolicy(sizePolicy)
        self.label_calibration_start_current.setObjectName("label_calibration_start_current")
        self.gridLayout.addWidget(self.label_calibration_start_current, 5, 0, 1, 1)
        self.calibration_start_current = QtWidgets.QLineEdit(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.calibration_start_current.sizePolicy().hasHeightForWidth())
        self.calibration_start_current.setSizePolicy(sizePolicy)
        self.calibration_start_current.setMinimumSize(QtCore.QSize(100, 20))
        self.calibration_start_current.setStyleSheet("QLineEdit{\n"
                                                "                                                background: rgb(223,223,233)\n"
                                                "                                                }\n"
                                                "                                            ")
        self.calibration_start_current.setObjectName("calibration_start_current")
        self.gridLayout.addWidget(self.calibration_start_current, 5, 1, 1, 1)
        self.label_179 = QtWidgets.QLabel(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_179.sizePolicy().hasHeightForWidth())
        self.label_179.setSizePolicy(sizePolicy)
        self.label_179.setObjectName("label_179")
        self.gridLayout.addWidget(self.label_179, 6, 0, 1, 1)
        self.measurement_conversion_mode = QtWidgets.QComboBox(parent=self.centralwidget)
        self.measurement_conversion_mode.setMinimumSize(QtCore.QSize(100, 20))
        self.measurement_conversion_mode.setStyleSheet("QComboBox{\n"
                                                       "                                                background: rgb(223,223,233)\n"
                                                       "                                                }\n"
                                                       "                                            ")
        self.measurement_conversion_mode.setObjectName("measurement_conversion_mode")
        self.measurement_conversion_mode.addItem("")
        self.measurement_conversion_mode.addItem("")
        self.gridLayout.addWidget(self.measurement_conversion_mode, 6, 1, 1, 1)
        self.label_resistivity_mode = QtWidgets.QLabel(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_resistivity_mode.sizePolicy().hasHeightForWidth())
        self.label_resistivity_mode.setSizePolicy(sizePolicy)
        self.label_resistivity_mode.setObjectName("label_resistivity_mode")
        self.gridLayout.addWidget(self.label_resistivity_mode, 7, 0, 1, 1)
        self.resistivity_measurement_mode = QtWidgets.QComboBox(parent=self.centralwidget)
        self.resistivity_measurement_mode.setMinimumSize(QtCore.QSize(100, 20))
        self.resistivity_measurement_mode.setStyleSheet("QComboBox{ background: rgb(223,223,233) }")
        self.resistivity_measurement_mode.setObjectName("resistivity_measurement_mode")
        for _ in tds_experiment.RESISTIVITY_MODES:
            self.resistivity_measurement_mode.addItem("")
        self.gridLayout.addWidget(self.resistivity_measurement_mode, 7, 1, 1, 1)
        self.label_material_profile = QtWidgets.QLabel(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_material_profile.sizePolicy().hasHeightForWidth())
        self.label_material_profile.setSizePolicy(sizePolicy)
        self.label_material_profile.setObjectName("label_material_profile")
        self.gridLayout.addWidget(self.label_material_profile, 8, 0, 1, 1)
        self.material_profile_row = QtWidgets.QWidget(parent=self.centralwidget)
        self.material_profile_row.setObjectName("material_profile_row")
        self.material_profile_layout = QtWidgets.QHBoxLayout(self.material_profile_row)
        self.material_profile_layout.setContentsMargins(0, 0, 0, 0)
        self.material_profile_layout.setSpacing(4)
        self.material_profile_combo = QtWidgets.QComboBox(parent=self.material_profile_row)
        self.material_profile_combo.setEditable(True)
        self.material_profile_combo.setMinimumSize(QtCore.QSize(70, 20))
        self.material_profile_combo.setStyleSheet("QComboBox{ background: rgb(223,223,233) }")
        self.material_profile_combo.setObjectName("material_profile_combo")
        self.material_profile_layout.addWidget(self.material_profile_combo)
        self.material_profile_load_button = QtWidgets.QPushButton(parent=self.material_profile_row)
        self.material_profile_load_button.setMinimumSize(QtCore.QSize(50, 20))
        self.material_profile_load_button.setStyleSheet("QPushButton{background: rgb(193, 193, 193)}")
        self.material_profile_load_button.setObjectName("material_profile_load_button")
        self.material_profile_layout.addWidget(self.material_profile_load_button)
        self.material_profile_save_button = QtWidgets.QPushButton(parent=self.material_profile_row)
        self.material_profile_save_button.setMinimumSize(QtCore.QSize(50, 20))
        self.material_profile_save_button.setStyleSheet("QPushButton{background: rgb(193, 193, 193)}")
        self.material_profile_save_button.setObjectName("material_profile_save_button")
        self.material_profile_layout.addWidget(self.material_profile_save_button)
        self.gridLayout.addWidget(self.material_profile_row, 8, 1, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        ###
        # self.temperature_vis = QtWidgets.QGraphicsView(parent=self.centralwidget)
        self.temperature_vis = pg.PlotWidget(parent=self.centralwidget)
        self.temperature_vis.setBackground('w')
        ###
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                           QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.temperature_vis.sizePolicy().hasHeightForWidth())
        self.temperature_vis.setSizePolicy(sizePolicy)
        self.temperature_vis.setMinimumSize(QtCore.QSize(260, 190))
        self.temperature_vis.setStyleSheet("QWidget{\n"
                                           "                                            border: 0.5px solid gray;\n"
                                           "                                            }\n"
                                           "                                        ")
        self.temperature_vis.setObjectName("temperature_vis")
        self.gridLayout_4.addWidget(self.temperature_vis, 0, 0, 1, 1)
        ###
        self.resistivity_vis = pg.PlotWidget(parent=self.centralwidget)
        self.resistivity_vis.setBackground('w')
        ###
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                           QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.resistivity_vis.sizePolicy().hasHeightForWidth())
        self.resistivity_vis.setSizePolicy(sizePolicy)
        self.resistivity_vis.setMinimumSize(QtCore.QSize(260, 190))
        self.resistivity_vis.setStyleSheet("QWidget{\n"
                                           "                                            border: 0.5px solid gray;\n"
                                           "                                            }\n"
                                           "                                        ")
        self.resistivity_vis.setObjectName("resistivity_vis")
        self.gridLayout_4.addWidget(self.resistivity_vis, 1, 0, 1, 1)
        ###
        # self.h_flux_vis = QtWidgets.QGraphicsView(parent=self.centralwidget)
        self.h_flux_vis = pg.PlotWidget(parent=self.centralwidget)
        self.h_flux_vis.setBackground('w')
        ###
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                           QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.h_flux_vis.sizePolicy().hasHeightForWidth())
        self.h_flux_vis.setSizePolicy(sizePolicy)
        self.h_flux_vis.setMinimumSize(QtCore.QSize(260, 190))
        self.h_flux_vis.setStyleSheet("QWidget{\n"
                                      "                                            border: 0.5px solid gray;\n"
                                      "                                            }\n"
                                      "                                        ")
        self.h_flux_vis.setObjectName("h_flux_vis")
        self.gridLayout_4.addWidget(self.h_flux_vis, 2, 0, 1, 1)
        self.plot_window_layout = QtWidgets.QHBoxLayout()
        self.plot_window_layout.setObjectName("plot_window_layout")
        self.plot_window_button = QtWidgets.QPushButton(parent=self.centralwidget)
        self.plot_window_button.setMinimumSize(QtCore.QSize(120, 24))
        self.plot_window_button.setMaximumSize(QtCore.QSize(140, 16777215))
        self.plot_window_button.setCheckable(True)
        self.plot_window_button.setStyleSheet(
            "QPushButton{background: rgb(193, 193, 193)}\n"
            "QPushButton:checked{background: rgb(190, 255, 190)}"
        )
        self.plot_window_button.setObjectName("plot_window_button")
        self.plot_window_layout.addStretch(1)
        self.plot_window_layout.addWidget(self.plot_window_button)
        self.plot_window_layout.addStretch(1)
        self.gridLayout_4.addLayout(self.plot_window_layout, 3, 0, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_4, 0, 1, 2, 1)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_4 = QtWidgets.QLabel(parent=self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout_3.addWidget(self.label_4, 0, 0, 1, 1)
        self.label_1 = QtWidgets.QLabel(parent=self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_1.setFont(font)
        self.label_1.setObjectName("label_1")
        self.gridLayout_3.addWidget(self.label_1, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(parent=self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 0, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(parent=self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 0, 3, 1, 1)
        self.label_5 = QtWidgets.QLabel(parent=self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 0, 4, 1, 1)
        self.temperature_target_lcd = QtWidgets.QLCDNumber(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                           QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.temperature_target_lcd.sizePolicy().hasHeightForWidth())
        self.temperature_target_lcd.setSizePolicy(sizePolicy)
        self.temperature_target_lcd.setMinimumSize(QtCore.QSize(100, 50))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.temperature_target_lcd.setFont(font)
        self.temperature_target_lcd.setStyleSheet("QLCDNumber{\n"
                                                  "                                            border: 2px solid red;\n"
                                                  "                                                            border-radius: 10px;\n"
                                                  "                                                            padding: 0 8px;\n"
                                                  "                                                            }\n"
                                                  "                                                        ")
        self.temperature_target_lcd.setObjectName("temperature_target_lcd")
        self.gridLayout_3.addWidget(self.temperature_target_lcd, 1, 0, 1, 1)
        self.temperature_lcd = QtWidgets.QLCDNumber(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                           QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.temperature_lcd.sizePolicy().hasHeightForWidth())
        self.temperature_lcd.setSizePolicy(sizePolicy)
        self.temperature_lcd.setMinimumSize(QtCore.QSize(100, 50))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.temperature_lcd.setFont(font)
        self.temperature_lcd.setStyleSheet("QLCDNumber{\n"
                                             "                                            border: 2px solid red;\n"
                                             "                                                            border-radius: 10px;\n"
                                             "                                                            padding: 0 8px;\n"
                                             "                                                            }\n"
                                             "                                                        ")
        self.temperature_lcd.setObjectName("temperature_lcd")
        self.gridLayout_3.addWidget(self.temperature_lcd, 1, 1, 1, 1)
        self.voltage_lcd = QtWidgets.QLCDNumber(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                           QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.voltage_lcd.sizePolicy().hasHeightForWidth())
        self.voltage_lcd.setSizePolicy(sizePolicy)
        self.voltage_lcd.setMinimumSize(QtCore.QSize(100, 50))
        self.voltage_lcd.setStyleSheet("QLCDNumber{\n"
                                       "                                            border: 2px solid orange;\n"
                                       "                                            border-radius: 10px;\n"
                                       "                                            padding: 0 8px;\n"
                                       "                                            }\n"
                                       "                                        ")
        self.voltage_lcd.setObjectName("voltage_lcd")
        self.gridLayout_3.addWidget(self.voltage_lcd, 1, 2, 1, 1)
        self.current_lcd = QtWidgets.QLCDNumber(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                           QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.current_lcd.sizePolicy().hasHeightForWidth())
        self.current_lcd.setSizePolicy(sizePolicy)
        self.current_lcd.setMinimumSize(QtCore.QSize(100, 50))
        self.current_lcd.setStyleSheet("QLCDNumber{\n"
                                       "                                            border: 2px solid blue;\n"
                                       "                                            border-radius: 10px;\n"
                                       "                                            padding: 0 8px;\n"
                                       "                                            }\n"
                                       "                                        ")
        self.current_lcd.setObjectName("current_lcd")
        self.gridLayout_3.addWidget(self.current_lcd, 1, 3, 1, 1)
        self.resistivity_lcd = QtWidgets.QLCDNumber(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                           QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.resistivity_lcd.sizePolicy().hasHeightForWidth())
        self.resistivity_lcd.setSizePolicy(sizePolicy)
        self.resistivity_lcd.setMinimumSize(QtCore.QSize(110, 50))
        self.resistivity_lcd.setStyleSheet("QLCDNumber{\n"
                                           "                                            border: 2px solid green;\n"
                                           "                                            border-radius: 10px;\n"
                                           "                                            padding: 0 8px;\n"
                                           "                                            }\n"
                                           "                                        ")
        self.resistivity_lcd.setObjectName("resistivity_lcd")
        self.gridLayout_3.addWidget(self.resistivity_lcd, 1, 4, 1, 1)
        self.label_power = QtWidgets.QLabel(parent=self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_power.setFont(font)
        self.label_power.setObjectName("label_power")
        self.gridLayout_3.addWidget(self.label_power, 0, 5, 1, 1)
        self.power_lcd = QtWidgets.QLCDNumber(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                           QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.power_lcd.sizePolicy().hasHeightForWidth())
        self.power_lcd.setSizePolicy(sizePolicy)
        self.power_lcd.setMinimumSize(QtCore.QSize(110, 50))
        self.power_lcd.setStyleSheet("QLCDNumber{\n"
                                     "                                            border: 2px solid purple;\n"
                                     "                                            border-radius: 10px;\n"
                                     "                                            padding: 0 8px;\n"
                                     "                                            }\n"
                                     "                                        ")
        self.power_lcd.setObjectName("power_lcd")
        self.gridLayout_3.addWidget(self.power_lcd, 1, 5, 1, 1)
        self.parameters_text = QtWidgets.QTextEdit(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                           QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.parameters_text.sizePolicy().hasHeightForWidth())
        self.parameters_text.setSizePolicy(sizePolicy)
        self.parameters_text.setMinimumSize(QtCore.QSize(0, 100))
        self.parameters_text.setStyleSheet(
            "QWidget{border: 2px solid gray; border-radius: 10px;padding: 0 8px; background: rgb(223,223,233)}\n"
            "                                    ")
        self.parameters_text.setObjectName("parameters_text")
        self.gridLayout_3.addWidget(self.parameters_text, 2, 0, 1, 6)
        self.pid_status_widget = QtWidgets.QWidget(parent=self.centralwidget)
        self.pid_status_widget.setObjectName("pid_status_widget")
        self.pid_status_layout = QtWidgets.QHBoxLayout(self.pid_status_widget)
        self.pid_status_layout.setContentsMargins(0, 0, 0, 0)
        self.pid_status_layout.setSpacing(6)
        pid_field_style = (
            "QLineEdit{border: 1px solid gray; border-radius: 6px; padding: 2px 6px; "
            "background: rgb(255,255,255); font-family: 'Consolas','Courier New',monospace;}"
        )
        self.controller_mode_label = QtWidgets.QLabel(parent=self.pid_status_widget)
        self.controller_mode_label.setObjectName("controller_mode_label")
        self.pid_status_layout.addWidget(self.controller_mode_label)
        self.pid_status_layout.addSpacing(10)
        self.label_pid_kp = QtWidgets.QLabel("Kp", parent=self.pid_status_widget)
        self.pid_status_layout.addWidget(self.label_pid_kp)
        self.pid_kp_edit = QtWidgets.QLineEdit(parent=self.pid_status_widget)
        self.pid_kp_edit.setMinimumSize(QtCore.QSize(90, 22))
        self.pid_kp_edit.setStyleSheet(pid_field_style)
        self.pid_kp_edit.setObjectName("pid_kp_edit")
        self.pid_status_layout.addWidget(self.pid_kp_edit)
        self.label_pid_ki = QtWidgets.QLabel("Ki", parent=self.pid_status_widget)
        self.pid_status_layout.addWidget(self.label_pid_ki)
        self.pid_ki_edit = QtWidgets.QLineEdit(parent=self.pid_status_widget)
        self.pid_ki_edit.setMinimumSize(QtCore.QSize(90, 22))
        self.pid_ki_edit.setStyleSheet(pid_field_style)
        self.pid_ki_edit.setObjectName("pid_ki_edit")
        self.pid_status_layout.addWidget(self.pid_ki_edit)
        self.label_pid_kd = QtWidgets.QLabel("Kd", parent=self.pid_status_widget)
        self.pid_status_layout.addWidget(self.label_pid_kd)
        self.pid_kd_edit = QtWidgets.QLineEdit(parent=self.pid_status_widget)
        self.pid_kd_edit.setMinimumSize(QtCore.QSize(90, 22))
        self.pid_kd_edit.setStyleSheet(pid_field_style)
        self.pid_kd_edit.setObjectName("pid_kd_edit")
        self.pid_status_layout.addWidget(self.pid_kd_edit)
        self.pid_status_layout.addSpacing(14)
        self.label_step_up = QtWidgets.QLabel("Step Up (A)", parent=self.pid_status_widget)
        self.pid_status_layout.addWidget(self.label_step_up)
        self.max_current_step_up_edit = QtWidgets.QLineEdit(parent=self.pid_status_widget)
        self.max_current_step_up_edit.setMinimumSize(QtCore.QSize(80, 22))
        self.max_current_step_up_edit.setStyleSheet(pid_field_style)
        self.max_current_step_up_edit.setObjectName("max_current_step_up_edit")
        self.pid_status_layout.addWidget(self.max_current_step_up_edit)
        self.label_step_down = QtWidgets.QLabel("Step Down (A)", parent=self.pid_status_widget)
        self.pid_status_layout.addWidget(self.label_step_down)
        self.max_current_step_down_edit = QtWidgets.QLineEdit(parent=self.pid_status_widget)
        self.max_current_step_down_edit.setMinimumSize(QtCore.QSize(80, 22))
        self.max_current_step_down_edit.setStyleSheet(pid_field_style)
        self.max_current_step_down_edit.setObjectName("max_current_step_down_edit")
        self.pid_status_layout.addWidget(self.max_current_step_down_edit)
        self.pid_status_layout.addStretch(1)
        self.gridLayout_3.addWidget(self.pid_status_widget, 3, 0, 1, 6)
        self.pid_schedule_label = QtWidgets.QLabel(parent=self.centralwidget)
        self.pid_schedule_label.setMinimumSize(QtCore.QSize(0, 20))
        self.pid_schedule_label.setWordWrap(True)
        self.pid_schedule_label.setStyleSheet(
            "QLabel{color: #444444; font-family: 'Consolas','Courier New',monospace; font-size: 8pt;}"
        )
        self.pid_schedule_label.setObjectName("pid_schedule_label")
        self.gridLayout_3.addWidget(self.pid_schedule_label, 4, 0, 1, 6)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.start_botton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.start_botton.setMinimumSize(QtCore.QSize(100, 20))
        self.start_botton.setMaximumSize(QtCore.QSize(60, 16777215))
        self.start_botton.setStyleSheet("QPushButton{background: rgb(193, 193, 193)}")
        self.start_botton.setObjectName("start_botton")
        self.gridLayout_2.addWidget(self.start_botton, 0, 0, 1, 1)
        self.find_csv_botton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.find_csv_botton.setMinimumSize(QtCore.QSize(100, 20))
        self.find_csv_botton.setMaximumSize(QtCore.QSize(100, 16777215))
        self.find_csv_botton.setStyleSheet("QPushButton{background: rgb(193, 193, 193)}\n"
                                           "                                            ")
        self.find_csv_botton.setObjectName("find_csv_botton")
        self.gridLayout_2.addWidget(self.find_csv_botton, 0, 1, 1, 1)
        self.calibrate_botton_base_t = QtWidgets.QPushButton(parent=self.centralwidget)
        self.calibrate_botton_base_t.setMinimumSize(QtCore.QSize(100, 20))
        self.calibrate_botton_base_t.setMaximumSize(QtCore.QSize(100, 16777215))
        self.calibrate_botton_base_t.setStyleSheet("QPushButton{background: rgb(193, 193, 193)}")
        self.calibrate_botton_base_t.setObjectName("calibrate_botton_base_t")
        self.gridLayout_2.addWidget(self.calibrate_botton_base_t, 0, 2, 1, 1)
        self.stop_botton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.stop_botton.setMinimumSize(QtCore.QSize(100, 20))
        self.stop_botton.setMaximumSize(QtCore.QSize(60, 16777215))
        self.stop_botton.setStyleSheet("QPushButton{background: rgb(193, 193, 193)}")
        self.stop_botton.setObjectName("stop_botton")
        self.gridLayout_2.addWidget(self.stop_botton, 1, 0, 1, 1)
        self.load_csv_botton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.load_csv_botton.setMinimumSize(QtCore.QSize(100, 20))
        self.load_csv_botton.setMaximumSize(QtCore.QSize(100, 16777215))
        self.load_csv_botton.setStyleSheet("QPushButton{background: rgb(193, 193, 193)}")
        self.load_csv_botton.setObjectName("load_csv_botton")
        self.gridLayout_2.addWidget(self.load_csv_botton, 1, 1, 1, 1)
        self.calibrate_botton_pid = QtWidgets.QPushButton(parent=self.centralwidget)
        self.calibrate_botton_pid.setMinimumSize(QtCore.QSize(100, 20))
        self.calibrate_botton_pid.setMaximumSize(QtCore.QSize(100, 16777215))
        self.calibrate_botton_pid.setStyleSheet("QPushButton{background: rgb(193, 193, 193)}")
        self.calibrate_botton_pid.setObjectName("calibrate_botton_pid")
        self.gridLayout_2.addWidget(self.calibrate_botton_pid, 1, 2, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_2, 5, 0, 1, 6)
        self.gridLayout_5.addLayout(self.gridLayout_3, 1, 0, 1, 1)
        self.Error = QtWidgets.QLabel(parent=self.centralwidget)
        self.Error.setMinimumSize(QtCore.QSize(1000, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setStrikeOut(False)
        self.Error.setFont(font)
        self.Error.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.Error.setWordWrap(True)
        self.Error.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse)
        self.Error.setObjectName("Error")
        self.gridLayout_5.addWidget(self.Error, 2, 0, 1, 2)
        self.gridLayout_5.setColumnStretch(0, 3)
        self.gridLayout_5.setColumnStretch(1, 5)
        self.gridLayout_5.setRowStretch(0, 5)
        self.gridLayout_5.setRowStretch(1, 3)
        self.gridLayout_6.addLayout(self.gridLayout_5, 0, 0, 1, 1)
        TDS.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=TDS)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1020, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(parent=self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuHelp = QtWidgets.QMenu(parent=self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        TDS.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=TDS)
        self.statusbar.setObjectName("statusbar")
        TDS.setStatusBar(self.statusbar)
        self.actionExit = QtGui.QAction(parent=TDS)
        self.actionExit.setObjectName("actionExit")
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(TDS)
        QtCore.QMetaObject.connectSlotsByName(TDS)
        TDS.setTabOrder(self.ex_number, self.ex_name)
        TDS.setTabOrder(self.ex_name, self.calib_temperature)
        TDS.setTabOrder(self.calib_temperature, self.max_power)
        TDS.setTabOrder(self.max_power, self.max_current)
        TDS.setTabOrder(self.max_current, self.calibration_start_current)
        TDS.setTabOrder(self.calibration_start_current, self.measurement_conversion_mode)
        TDS.setTabOrder(self.measurement_conversion_mode, self.parameters_text)
        TDS.setTabOrder(self.parameters_text, self.start_botton)
        TDS.setTabOrder(self.start_botton, self.stop_botton)
        TDS.setTabOrder(self.stop_botton, self.find_csv_botton)
        TDS.setTabOrder(self.find_csv_botton, self.load_csv_botton)
        TDS.setTabOrder(self.load_csv_botton, self.calibrate_botton_base_t)
        TDS.setTabOrder(self.calibrate_botton_base_t, self.calibrate_botton_pid)
        TDS.setTabOrder(self.calibrate_botton_pid, self.temperature_vis)
        TDS.setTabOrder(self.temperature_vis, self.resistivity_vis)
        TDS.setTabOrder(self.resistivity_vis, self.h_flux_vis)

        #####
        self.timer_error = QtCore.QTimer()
        self.timer_error.timeout.connect(self.hideMessage)

        self.emitter.experiment_signal.connect(self.update_experiment_signal)
        self.emitter.live_measurement_signal.connect(self.update_live_measurement)
        self.emitter.calibration_warning_signal.connect(self.show_calibration_warning)

        self.find_csv_botton.clicked.connect(self.find_csv_clicked)
        self.load_csv_botton.clicked.connect(self.load_csv_clicked)
        self.start_botton.clicked.connect(self.start_clicked)
        self.stop_botton.clicked.connect(self.stop_clicked)
        self.calibrate_botton_base_t.clicked.connect(self.calibrate_base_temperature)
        self.calibrate_botton_pid.clicked.connect(self.calibrate_pid)
        self.plot_window_button.toggled.connect(self.toggle_plot_window)
        self.calibrate_botton_pid.setEnabled(True)

        self.h_flux_x = [i * 0.5 for i in range(200)]
        self.h_flux_y = [0.0] * 200
        self.h_flux_y = [np.nan] * len(self.h_flux_x)
        pen_h_flux = pg.mkPen(color=(0, 0, 225), width=4)
        self.h_flux_vis_line = self.h_flux_vis.plot(self.h_flux_x, self.h_flux_y, pen=pen_h_flux)

        self.resistivity_x = [i * 0.5 for i in range(200)]
        self.resistivity_y = [np.nan] * len(self.resistivity_x)
        pen_resistivity = pg.mkPen(color=(0, 150, 0), width=4)
        self.resistivity_vis_line = self.resistivity_vis.plot(self.resistivity_x, self.resistivity_y, pen=pen_resistivity)

        self.temperature_x = [i * 0.5 for i in range(200)]
        self.temperature_y = [0.0] * 200
        self.temperature_y = [np.nan] * len(self.temperature_x)
        self.temperature_y_target = [np.nan] * len(self.temperature_x)
        pen_temperature = pg.mkPen(color=(225, 0, 0, 90), width=4)
        pen_temperature_target = pg.mkPen(color=(0, 0, 225), width=3, style=QtCore.Qt.PenStyle.DashLine)
        self.temperature_vis_line_target = self.temperature_vis.plot(self.temperature_x, self.temperature_y_target,
                                                                     pen=pen_temperature_target)
        self.temperature_vis_line = self.temperature_vis.plot(self.temperature_x, self.temperature_y,
                                                              pen=pen_temperature)
        self.legend = pg.LegendItem(offset=(-5, -35))  # Position legend
        self.legend.setParentItem(self.temperature_vis.getPlotItem())  # Attach to the plot
        self.legend.addItem(self.temperature_vis_line, f"Diff: "
                                                       f"{abs(self.target_temperature - self.temperature):.2f} C")

        self.diff_label = self.legend.items[0][1]
        self.diff_label.setText(
            f"Diff: {abs(self.target_temperature - self.temperature):.2f} C",
            color="#000000",
        )

        # Add Axis Labels
        self.styles = {"color": "#f00", "font-size": "12px"}
        self.temperature_vis.setLabel("left", "Temperature", units="C", **self.styles)
        self.temperature_vis.setLabel("bottom", "Time (s)", **self.styles)
        self.resistivity_vis.setLabel("left", "Resistivity", units="Ohm", **self.styles)
        self.resistivity_vis.setLabel("bottom", "Time (s)", **self.styles)
        self.h_flux_vis.setLabel("left", "Flux", units='mol/s', **self.styles)
        self.h_flux_vis.setLabel("bottom", "Time (s)", **self.styles)

        self.temperature_vis.showGrid(x=True, y=True)
        self.resistivity_vis.showGrid(x=True, y=True)
        self.h_flux_vis.showGrid(x=True, y=True)
        self.refresh_plot_ranges()

        self.update_timer = QTimer()  # Create a QTimer for updating graphs
        # self.update_timer.start(500)
        self.update_timer.timeout.connect(self.update_graphs)  # Connect it to the update_graphs slot

        self.ex_number.setEnabled(False)
        self.ex_number.setText(str(self.ex_counter))

        self.max_power.editingFinished.connect(self.update_max_power)
        self.max_current.editingFinished.connect(self.update_max_current)
        self.calibration_start_current.editingFinished.connect(self.update_calibration_start_current)
        self.pid_kp_edit.editingFinished.connect(self.update_pid_kp)
        self.pid_ki_edit.editingFinished.connect(self.update_pid_ki)
        self.pid_kd_edit.editingFinished.connect(self.update_pid_kd)
        self.max_current_step_up_edit.editingFinished.connect(self.update_max_current_step_up)
        self.max_current_step_down_edit.editingFinished.connect(self.update_max_current_step_down)
        self.material_profile_load_button.clicked.connect(self.load_material_profile)
        self.material_profile_save_button.clicked.connect(self.save_material_profile)
        self.measurement_conversion_mode.currentIndexChanged.connect(self.update_experiment_mode)
        self.resistivity_measurement_mode.currentIndexChanged.connect(self.update_resistivity_mode)
        self.calib_temperature.textEdited.connect(self.invalidate_t_zero_calibration)

        # Populate inputs from configuration.
        self.max_power.setText(str(self.config['max_power_w']))
        self.max_current.setText(str(self.config['max_current']))
        self.calibration_start_current.setText(str(self.config['t0_current_search_start']))
        self.measurement_conversion_mode.setCurrentText(
            tds_experiment.get_experiment_mode(self.config)
        )
        self.resistivity_measurement_mode.setCurrentText(
            tds_experiment.get_resistivity_mode(self.config)
        )
        self.apply_experiment_mode_ui()
        self._update_file_tooltips()
        self._set_operation_running(False)
        self.refresh_pid_status_label()
        self.refresh_material_profile_list()

        self.voltage_lcd.setDigitCount(8)
        self.current_lcd.setDigitCount(8)
        self.resistivity_lcd.setDigitCount(8)
        self.power_lcd.setDigitCount(8)
        self.temperature_lcd.setDigitCount(6)
        self.temperature_target_lcd.setDigitCount(6)

    def retranslateUi(self, TDS):
        _translate = QtCore.QCoreApplication.translate
        TDS.setWindowTitle(_translate("TDS", "TDS"))
        self.label_183.setText(_translate("TDS", "Experiment Number"))
        self.ex_number.setText(_translate("TDS", "1"))
        self.label_175.setText(_translate("TDS", "Experiment Name"))
        self.ex_name.setText(_translate("TDS", "test"))
        self.label_176.setText(_translate("TDS", "Zero Temperature (°C)"))
        self.calib_temperature.setText(_translate("TDS", "23"))
        self.label_177.setText(_translate("TDS", "Max Power (W)"))
        self.max_power.setText(_translate("TDS", "10"))
        self.label_178.setText(_translate("TDS", "Max Current (A)"))
        self.max_current.setText(_translate("TDS", "1"))
        self.label_calibration_start_current.setText(_translate("TDS", "Initial Current (A)"))
        self.calibration_start_current.setText(_translate("TDS", "0.01"))
        self.label_179.setText(_translate("TDS", "Mode"))
        self.measurement_conversion_mode.setItemText(0, _translate("TDS", "TEMPERATURE"))
        self.measurement_conversion_mode.setItemText(1, _translate("TDS", "CURRENT"))
        self.label_resistivity_mode.setText(_translate("TDS", "Resistivity Mode"))
        for index, mode in enumerate(tds_experiment.RESISTIVITY_MODES):
            self.resistivity_measurement_mode.setItemText(index, _translate("TDS", mode))
        self.resistivity_measurement_mode.setToolTip(_translate("TDS", RESISTIVITY_MODE_TOOLTIP))
        self.label_material_profile.setText(_translate("TDS", "Material Profile"))
        self.material_profile_combo.setToolTip(_translate(
            "TDS", "Saved per-material tuning result: gain schedule, step limits, and related settings."
        ))
        self.material_profile_load_button.setText(_translate("TDS", "Load"))
        self.material_profile_save_button.setText(_translate("TDS", "Save"))
        self.label_4.setText(_translate("TDS", "Target Temp. (°C)    "))
        self.label_1.setText(_translate("TDS", "Measured Temp (°C)"))
        self.label_2.setText(_translate("TDS", "Voltage (V)              "))
        self.label_3.setText(_translate("TDS", "Current (A)                "))
        self.label_5.setText(_translate("TDS", "Resistivity (Ohm)"))
        self.label_power.setText(_translate("TDS", "Power (W)"))
        self.controller_mode_label.setText(_translate("TDS", "Controller: PI"))
        self.parameters_text.setHtml(_translate("TDS",
                                                "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                                "<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
                                                "p, li { white-space: pre-wrap; }\n"
                                                "</style></head><body style=\" font-family:\'Segoe UI\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
                                                "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'JetBrains Mono,monospace\'; font-size:8pt; color:#000000;\">{start_T=40;step_T</span><span style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.875pt;\">=200;target_T=200;ramp_speed_min=10;hold_step_time_min=1</span><span style=\" font-family:\'JetBrains Mono,monospace\'; font-size:8pt; color:#000000;\">}</span>                                                                              </p></body></html>"))
        self.start_botton.setText(_translate("TDS", "Start"))
        self.find_csv_botton.setText(_translate("TDS", "Find R vs. T"))
        self.calibrate_botton_base_t.setText(_translate("TDS", "Calibrate T. Zero"))
        self.stop_botton.setText(_translate("TDS", "Stop"))
        self.load_csv_botton.setText(_translate("TDS", "Reload R vs. T"))
        self.calibrate_botton_pid.setText(_translate("TDS", "Tune PI/PID"))
        self.plot_window_button.setText(_translate("TDS", "Last 60 s"))
        self.Error.setText(_translate("TDS", "<html><head/><body><p><br/></p></body></html>"))
        self.menuFile.setTitle(_translate("TDS", "File"))
        self.menuHelp.setTitle(_translate("TDS", "Help"))
        self.actionExit.setText(_translate("TDS", "Exit"))

    def update_max_power(self):
        """Validate and save the maximum measured sample power."""
        previous_value = float(self.config['max_power_w'])
        try:
            max_power_w = float(self.max_power.text())
            if not np.isfinite(max_power_w) or max_power_w <= 0:
                raise ValueError('Max Power must be positive and finite.')
        except ValueError as exc:
            self.max_power.setText(f'{previous_value:g}')
            self.error_message(str(exc), color='red')
            return False

        self.config['max_power_w'] = max_power_w
        self.max_power.setText(f'{max_power_w:g}')
        self.emitter.max_power_signal.emit(max_power_w)
        self.save_config()
        return True

    def update_max_current(self):
        """Update the highest current the controller may command."""
        self.config['max_current'] = float(self.max_current.text())
        self.emitter.max_current_signal.emit(self.config['max_current'])
        self.save_config()

    def update_calibration_start_current(self):
        """Validate and save the shared initial current used by all operations."""
        previous_value = float(self.config['t0_current_search_start'])
        try:
            current = float(self.calibration_start_current.text())
            if not np.isfinite(current) or current < 0.001:
                raise ValueError(
                    'Initial current must be at least 0.001 A; the supply programs current in 1 mA steps.'
                )
            if current > float(self.config['max_current']):
                raise ValueError('Initial current exceeds Max Current.')
        except ValueError as exc:
            self.calibration_start_current.setText(f'{previous_value:g}')
            self.error_message(str(exc), color='red')
            return False

        self.config['t0_current_search_start'] = current
        self.config['tuning_start_current'] = current
        self.config['startup_current'] = current
        self.calibration_start_current.setText(f'{current:g}')
        self.save_config()
        return True

    def update_experiment_mode(self):
        """
        Update the selected experiment mode.
        """
        if self._active_experiment_mode is not None:
            self._program_text_by_mode[self._active_experiment_mode] = self.parameters_text.toPlainText()
        self.config['experiment_mode'] = self.measurement_conversion_mode.currentText()
        self.apply_experiment_mode_ui()
        self.save_config()

    def update_resistivity_mode(self):
        """Update how sample resistance is measured."""
        self.config['resistivity_mode'] = self.resistivity_measurement_mode.currentText()
        self.save_config()

    def refresh_pid_status_label(self):
        """Show the controller gains/step limits currently in use, above
        Start/Stop, and let the user edit them directly.

        Reflects self.config, so it updates immediately after a tune (or a
        config/profile load) rather than only showing whatever was tuned last.
        """
        controller_mode = tds_experiment.get_controller_mode(self.config)
        schedule = self.config.get('pid_gain_schedule') or []
        kp = float(self.config.get('pid_kp', 0.0))
        ki = float(self.config.get('pid_ki', 0.0))
        kd = float(self.config.get('pid_kd', 0.0))
        mode_text = f"Controller: {controller_mode}"
        if schedule:
            mode_text += f" ({len(schedule)}-point schedule)"
        self.controller_mode_label.setText(mode_text)
        if not self.pid_kp_edit.hasFocus():
            self.pid_kp_edit.setText(f'{kp:g}')
        if not self.pid_ki_edit.hasFocus():
            self.pid_ki_edit.setText(f'{ki:g}')
        if not self.pid_kd_edit.hasFocus():
            self.pid_kd_edit.setText(f'{kd:g}')
        kd_used = controller_mode == 'PID'
        self.pid_kd_edit.setEnabled(kd_used)
        self.label_pid_kd.setEnabled(kd_used)
        self.pid_kd_edit.setToolTip('' if kd_used else 'Kd is only used when controller_mode is PID.')

        step_up = float(self.config.get('max_current_step_up', 0.0))
        step_down = float(self.config.get('max_current_step_down', 0.0))
        if not self.max_current_step_up_edit.hasFocus():
            self.max_current_step_up_edit.setText(f'{step_up:g}')
        if not self.max_current_step_down_edit.hasFocus():
            self.max_current_step_down_edit.setText(f'{step_down:g}')

        if schedule:
            points_text = "  |  ".join(
                f"{point['current_a']:.4g} A: Kp={point['kp']:.4g} Ki={point['ki']:.4g}"
                + (f" Kd={point['kd']:.4g}" if kd_used else "")
                for point in sorted(schedule, key=lambda point: point['current_a'])
            )
            self.pid_schedule_label.setText(f"Gain schedule - {points_text}")
        else:
            self.pid_schedule_label.setText(
                "No gain schedule - Kp/Ki/Kd above apply at every current. Tune PI/PID to build one."
            )

    def _update_pid_gain(self, line_edit, config_key, label):
        """Validate and save one PID/PI gain edited directly in the GUI.

        Editing a flat gain by hand clears any multi-point schedule, since a
        schedule would otherwise silently override this value during a run.
        """
        previous_value = float(self.config.get(config_key, 0.0))
        try:
            value = float(line_edit.text())
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{label} must be zero or a positive finite number.')
        except ValueError as exc:
            line_edit.setText(f'{previous_value:g}')
            self.error_message(str(exc), color='red')
            return False

        self.config[config_key] = value
        if self.config.get('pid_gain_schedule'):
            self.config['pid_gain_schedule'] = []
            self.error_message(
                f'{label} set manually; the multi-point gain schedule was cleared.', color='black'
            )
        line_edit.setText(f'{value:g}')
        self.save_config()
        self.refresh_pid_status_label()
        return True

    def update_pid_kp(self):
        """Save a manually edited proportional gain."""
        return self._update_pid_gain(self.pid_kp_edit, 'pid_kp', 'Kp')

    def update_pid_ki(self):
        """Save a manually edited integral gain."""
        return self._update_pid_gain(self.pid_ki_edit, 'pid_ki', 'Ki')

    def update_pid_kd(self):
        """Save a manually edited derivative gain."""
        return self._update_pid_gain(self.pid_kd_edit, 'pid_kd', 'Kd')

    def _update_current_step(self, line_edit, config_key, label):
        """Validate and save one current-per-loop step limit."""
        previous_value = float(self.config.get(config_key, 0.01))
        try:
            value = float(line_edit.text())
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{label} must be a positive finite number of amps.')
        except ValueError as exc:
            line_edit.setText(f'{previous_value:g}')
            self.error_message(str(exc), color='red')
            return False

        self.config[config_key] = value
        line_edit.setText(f'{value:g}')
        self.save_config()
        return True

    def update_max_current_step_up(self):
        """Save a manually edited upward per-loop current step limit."""
        return self._update_current_step(self.max_current_step_up_edit, 'max_current_step_up', 'Step Up')

    def update_max_current_step_down(self):
        """Save a manually edited downward per-loop current step limit."""
        return self._update_current_step(
            self.max_current_step_down_edit, 'max_current_step_down', 'Step Down'
        )

    def refresh_material_profile_list(self):
        """Repopulate the saved-profile dropdown from files/material_profiles/."""
        current_text = self.material_profile_combo.currentText()
        self.material_profile_combo.blockSignals(True)
        try:
            self.material_profile_combo.clear()
            self.material_profile_combo.addItems(material_profiles.list_profiles())
            self.material_profile_combo.setCurrentText(current_text)
        finally:
            self.material_profile_combo.blockSignals(False)

    def save_material_profile(self):
        """Snapshot the current tuning result and related settings under a name."""
        name = self.material_profile_combo.currentText().strip()
        if not name:
            self.error_message('Enter a material name before saving a profile.', color='red')
            return
        try:
            path = material_profiles.save_profile(name, self.config)
        except ValueError as exc:
            self.error_message(str(exc), color='red')
            return
        self.refresh_material_profile_list()
        self.material_profile_combo.setCurrentText(name)
        self.error_message(f'Saved material profile "{name}" to {path.name}.', color='black')

    def load_material_profile(self):
        """Apply a saved profile's gain schedule/step limits/settings to the live config."""
        name = self.material_profile_combo.currentText().strip()
        if not name:
            self.error_message('Select a saved material profile to load.', color='red')
            return
        try:
            profile = material_profiles.load_profile(name)
        except FileNotFoundError as exc:
            self.error_message(str(exc), color='red')
            return

        for field in material_profiles.PROFILE_FIELDS:
            if field in profile:
                self.config[field] = profile[field]
        self.save_config()
        self.refresh_pid_status_label()
        self.resistivity_measurement_mode.setCurrentText(tds_experiment.get_resistivity_mode(self.config))
        self.max_current.setText(f"{float(self.config['max_current']):g}")
        self.error_message(f'Loaded material profile "{name}".', color='black')

    def apply_experiment_mode_ui(self):
        """Enable or disable controls based on the selected experiment mode."""
        if self._operation_running:
            return
        experiment_mode = tds_experiment.get_experiment_mode(self.config)
        current_mode = experiment_mode == "CURRENT"
        if experiment_mode != self._active_experiment_mode:
            self.parameters_text.setPlainText(self._program_text_by_mode[experiment_mode])
            self._active_experiment_mode = experiment_mode
        # T0 calibration anchors the loaded material curve to the current wire,
        # so it is required in both Temperature and Current modes.
        self.calib_temperature.setEnabled(True)
        self.calibrate_botton_base_t.setEnabled(True)
        self.calibrate_botton_pid.setEnabled(not current_mode)
        self.parameters_text.setEnabled(True)
        if current_mode:
            self.parameters_text.setPlaceholderText("{ramp_speed_min=0.01}  (A/min)")
            self.label_4.setText("PSU Command (V)")
        else:
            self.parameters_text.setPlaceholderText(
                "{start_T=40;step_T=200;target_T=200;ramp_speed_min=10;hold_step_time_min=1}"
            )
            self.label_4.setText("Target Temp. (°C)    ")

    def _set_operation_running(self, running):
        """Lock setup controls while hardware is calibrating, tuning, or running."""
        self._operation_running = bool(running)
        setup_controls = (
            self.ex_name,
            self.calib_temperature,
            self.max_power,
            self.max_current,
            self.calibration_start_current,
            self.measurement_conversion_mode,
            self.parameters_text,
        )
        action_controls = (
            self.start_botton,
            self.find_csv_botton,
            self.load_csv_botton,
            self.calibrate_botton_base_t,
            self.calibrate_botton_pid,
        )
        for control in (*setup_controls, *action_controls):
            control.setEnabled(not self._operation_running)
        self.stop_botton.setEnabled(self._operation_running)
        if not self._operation_running:
            self.apply_experiment_mode_ui()

    def _update_file_tooltips(self):
        """Describe the R-vs-T file actions and show the selected file on hover."""
        self.find_csv_botton.setToolTip(
            'Choose the material R-vs-T reference file (.xlsx or .csv).\n'
            'The file must contain resistivity and temperature columns, then run Calibrate T. Zero.'
        )
        selected_file = self.file_path or 'No R-vs-T file selected yet.'
        self.load_csv_botton.setToolTip(
            'Reload the currently selected R-vs-T reference file.\n'
            f'Selected file: {selected_file}'
        )
        self.calibration_start_current.setToolTip(
            'Shared starting PSU voltage and enforced experiment floor (default: 0.01 V).\n'
            'Startup requires stable readings and searches upward only in cautious 0.001 V steps.'
        )
        self.calibrate_botton_base_t.setToolTip(
            'Measure the current wire at the entered zero temperature and scale the loaded material curve.'
        )
        self.calibrate_botton_pid.setToolTip(
            'Tune the PI/PID controller after a successful T. Zero calibration.'
        )
        self.max_power.setToolTip(
            'Hard software cutoff for measured sample power |Vsample x I| (default: 2.5 W).\n'
            'For thin wires, choose a much lower value validated with an independent temperature measurement.'
        )

    def find_csv_clicked(self):
        """
        Opens a file dialog in a separate thread to avoid blocking the UI.
        """
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select File", "", "Excel Files (*.xlsx);;CSV Files (*.csv)"
        )
        if file_path:
            self.file_path = file_path
            self._update_file_tooltips()
            self.load_csv_clicked()

    def load_csv_clicked(self):
        """
        Load a resistance/resistivity-vs-temperature CSV or Excel file.
        """
        if not self.file_path:
            self.error_message('Select an R vs. T file first', color='red')
            return False
        try:
            curve, source = load_resistance_temperature_file(self.file_path)
        except Exception as exc:
            self.error_message(f'Failed to load R vs. T file: {exc}', color='red')
            print(f"Failed to load R vs. T file {self.file_path!r}: {exc}")
            return False

        self.r_vs_t = curve
        self.t_zero_calibrated = False
        self.t0_calibration_warning = None
        self.apply_experiment_mode_ui()
        print(
            f"Loaded R vs. T table from sheet {source['sheet']!r}, header row {source['header_row']}: "
            f"resistance={source['resistance_column']!r}, temperature={source['temperature_column']!r}, "
            f"points={source['points']}."
        )
        file_name = os.path.basename(self.file_path)
        if tds_experiment.get_experiment_mode(self.config) == "CURRENT":
            message = f'Loaded {file_name}. Run Calibrate T. Zero before starting Current mode.'
        else:
            message = f'Loaded {file_name}. Run Calibrate T. Zero before Tune PI/PID or Start.'
        self.error_message(message, color='black')
        return True

    def parse_experiment_params(self):
        """
        Parse the experiment program from the text box into a list of dictionaries.
        """
        parsed_params = []
        text = self.parameters_text.toPlainText().strip()
        if not text:
            raise ValueError('Experiment parameters are empty')

        required_keys = {'start_T', 'step_T', 'target_T', 'ramp_speed_min', 'hold_step_time_min'}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = line.strip("{}").strip()
            single_dict = {}
            for pair in line.split(";"):
                pair = pair.strip()
                if not pair:
                    continue
                if "=" not in pair:
                    raise ValueError(f'Invalid parameter entry: {pair}')
                key, value = pair.split("=", 1)
                single_dict[key.strip()] = float(value.strip(" }"))

            missing_keys = required_keys - set(single_dict)
            if missing_keys:
                missing_list = ", ".join(sorted(missing_keys))
                raise ValueError(f'Missing experiment parameters: {missing_list}')
            parsed_params.append(single_dict)

        if not parsed_params:
            raise ValueError('No experiment parameters found')
        return parsed_params

    def parse_voltage_ramp_params(self):
        """Parse the single open-loop voltage ramp rate in volts per minute."""
        text = self.parameters_text.toPlainText().strip().strip("{}").strip()
        if not text:
            raise ValueError('Current mode requires {ramp_speed_min=...}.')

        parsed = {}
        for pair in text.split(";"):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(f'Invalid voltage-mode parameter entry: {pair}')
            key, value = pair.split("=", 1)
            parsed[key.strip()] = float(value.strip(" }"))

        unexpected = set(parsed) - {'ramp_speed_min'}
        if unexpected:
            raise ValueError(
                'Current mode accepts only ramp_speed_min in A/min. '
                f'Unexpected parameters: {", ".join(sorted(unexpected))}'
            )
        if 'ramp_speed_min' not in parsed:
            raise ValueError('Current mode requires ramp_speed_min in A/min.')
        ramp_speed_min = float(parsed['ramp_speed_min'])
        if not np.isfinite(ramp_speed_min) or ramp_speed_min <= 0:
            raise ValueError('Voltage-mode ramp_speed_min must be positive and finite.')
        return {'ramp_speed_min': ramp_speed_min}

    def save_config(self):
        """
        Persist updated safety and PID settings to the local config file.

        pid_gain_schedule is a list of {current_a, kp, ki, kd} points, not a
        scalar, so it is not written into config.toml's flat key = value
        format. It lives in self.config for the running session and is
        persisted instead via a saved Material Profile.
        """
        config_io.save_config({key: value for key, value in self.config.items() if key != 'pid_gain_schedule'})

    def invalidate_t_zero_calibration(self):
        """
        Mark the current T0 calibration as outdated.
        """
        self.t_zero_calibrated = False

    def sanitize_experiment_name(self, name):
        """
        Convert the experiment name into a filesystem-safe folder name.
        """
        cleaned_name = re.sub(r'[<>:"/\\|?*]+', '_', name.strip())
        cleaned_name = cleaned_name.strip(' ._')
        return cleaned_name or 'TDS_test'

    def build_experiment_dir(self):
        """
        Build the output directory for the current experiment number and name.
        """
        experiment_name = self.sanitize_experiment_name(self.ex_name.text() or self.experiment_name)
        return DATA_DIR / f'{self.ex_counter}_{experiment_name}'

    def build_run_metadata(self):
        """Snapshot the controller/measurement settings used for this run.

        Excludes runtime-only keys (leading underscore, e.g. the active DMM
        range tracked in the config dict) since those are not settings.
        """
        return {
            key: value
            for key, value in self.config.items()
            if not str(key).startswith('_')
        }

    def can_close_window(self):
        """
        Ask the user for confirmation before closing the application.
        """
        if self.calibration_worker is not None and self.calibration_worker.isRunning():
            self.error_message('Wait until calibration or PI/PID tuning finishes before closing.', color='red')
            return False

        if self.worker_thread is not None and self.worker_thread.isRunning():
            reply = QtWidgets.QMessageBox.question(
                None,
                'Close TDS',
                'An experiment is running. Stop the experiment and close the application?',
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return False
            self.emitter.emit_stop()
            self.error_message('Stopping experiment before closing...', color='black')
            QtWidgets.QApplication.processEvents()
            if not self.worker_thread.wait(10000):
                self.error_message('Experiment is still stopping. Try closing again in a moment.', color='red')
                return False
            return True

        reply = QtWidgets.QMessageBox.question(
            None,
            'Close TDS',
            'Are you sure you want to close the application?',
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return reply == QtWidgets.QMessageBox.StandardButton.Yes

    def require_loaded_curve(self, action_name):
        """
        Ensure an R vs. T curve is loaded before continuing.
        """
        if self.r_vs_t is None:
            self.error_message(f'Load an R vs. T file before {action_name}.', color='red')
            return False
        return True

    def require_loaded_curve_and_t0(self, action_name):
        """
        Ensure the curve is loaded and T0 calibration succeeded.
        """
        if not self.require_loaded_curve(action_name):
            return False
        if not self.t_zero_calibrated:
            self.error_message(
                f'Run Calibrate T. Zero after loading R vs. T before {action_name}.',
                color='red',
            )
            return False
        return True

    def update_experiment_signal(self, data):
        """
        Updates the experiment signal with the new data

        Args:
            data (list): List of data to update the experiment signal

        Return:
            None
        """
        current_mode = tds_experiment.get_experiment_mode(self.config) == "CURRENT"
        self._apply_measurement_to_displays(
            target_temperature=None if current_mode else data[1],
            temperature=data[2],
            voltage=data[4],
            current=data[5],
            resistivity=data[8] if len(data) > 8 else None,
            power=data[7] if len(data) > 7 else None,
        )
        self.commanded_current = float(data[6]) if self._is_finite_number(data[6]) else 0.0
        if current_mode:
            self.temperature_target_lcd.display(self._format_lcd_value(self.commanded_current, decimals=4))

        # "time",  # Time in UNIX-readable format
        # "set_T",  # Set temperature
        # "T",  # Measured temperature
        # "h_f"  # Heat flux
        # "V",  # Voltage
        # "I",  # Current
        # "C_V",  # Calculated Power supply voltage
        # "P",  # Measured sample power
        # "R_ohm",  # Calculated Kelvin sample resistance
        self.data_list.append(list(data))

    def _is_finite_number(self, value):
        try:
            return bool(np.isfinite(value))
        except (TypeError, ValueError):
            return False

    def _format_lcd_value(self, value, decimals=4):
        if not self._is_finite_number(value):
            return "0"

        numeric_value = float(value)
        magnitude = abs(numeric_value)
        if magnitude == 0.0:
            return "0.0000"
        if 1e-3 <= magnitude < 1e3:
            return f"{numeric_value:.{decimals}f}"
        return f"{numeric_value:.2e}"

    def _apply_measurement_to_displays(
        self,
        *,
        target_temperature=None,
        temperature=None,
        voltage=None,
        current=None,
        resistivity=None,
        power=None,
    ):
        if self._is_finite_number(target_temperature):
            self.target_temperature = float(target_temperature)
        if self._is_finite_number(temperature):
            self.temperature = float(temperature)
        if self._is_finite_number(voltage):
            self.voltage = float(voltage)
        if self._is_finite_number(current):
            self.current = float(current)
        if self._is_finite_number(resistivity):
            self.resistivity = float(resistivity)
        elif (
            tds_experiment.get_resistivity_mode(self.config) == "V_OVER_I"
            and self._is_finite_number(self.voltage)
            and self._is_finite_number(self.current)
            and abs(self.current) > 1e-12
        ):
            corrected_resistance = float(self.voltage / self.current) - float(
                self.config.get("fixed_series_resistance_ohm", 0.0)
            )
            self.resistivity = corrected_resistance if np.isfinite(corrected_resistance) and corrected_resistance > 0 else 0

        if self._is_finite_number(power):
            self.sample_power = float(power)
        elif self._is_finite_number(self.voltage) and self._is_finite_number(self.current):
            self.sample_power = float(self.voltage) * float(self.current)

        if self._is_finite_number(self.voltage):
            self.voltage_lcd.display(self._format_lcd_value(self.voltage, decimals=4))
        else:
            self.voltage_lcd.display(0)

        if self._is_finite_number(self.current):
            self.current_lcd.display(self._format_lcd_value(self.current, decimals=4))
        else:
            self.current_lcd.display(0)

        if self._is_finite_number(self.resistivity):
            self.resistivity_lcd.display(self._format_lcd_value(self.resistivity, decimals=4))
        else:
            self.resistivity_lcd.display(0)

        if self._is_finite_number(self.sample_power):
            self.power_lcd.display(self._format_lcd_value(self.sample_power, decimals=4))
        else:
            self.power_lcd.display(0)

        if self._is_finite_number(self.temperature):
            self.temperature_lcd.display(round(self.temperature, 2))
        else:
            self.temperature_lcd.display(0)

        if self._is_finite_number(self.target_temperature):
            self.temperature_target_lcd.display(round(self.target_temperature, 2))
        else:
            self.temperature_target_lcd.display(0)

    def update_live_measurement(self, measurement):
        """
        Update the LCDs with live readings from T0 calibration and PID tuning.
        """
        if not isinstance(measurement, dict):
            return

        self._apply_measurement_to_displays(
            target_temperature=measurement.get("target_temperature"),
            temperature=measurement.get("temperature"),
            voltage=measurement.get("measured_voltage"),
            current=measurement.get("measured_current"),
            resistivity=measurement.get("resistance"),
        )

    def toggle_plot_window(self, checked):
        """
        Toggle between the full experiment view and the last 60 seconds.
        """
        self.plot_window_last60_selected = bool(checked)
        self.refresh_plot_ranges()

    def _current_plot_end_time(self):
        """
        Return the latest plotted experiment time in seconds.
        """
        if self.index_plot <= 0:
            return 0.0

        temperature_index = min(max(self.index_plot - 1, 0), len(self.temperature_x) - 1)
        h_flux_index = min(max(self.index_plot - 1, 0), len(self.h_flux_x) - 1)
        return float(max(self.temperature_x[temperature_index], self.h_flux_x[h_flux_index], 0.0))

    def refresh_plot_ranges(self):
        """
        Keep both plots on the selected time window.
        """
        end_time = self._current_plot_end_time()
        if self.plot_window_last60_selected:
            visible_end = max(self.plot_window_seconds, end_time)
            visible_start = max(0.0, visible_end - self.plot_window_seconds)
        else:
            visible_start = 0.0
            visible_end = max(1.0, end_time)

        for plot_widget in (self.temperature_vis, self.resistivity_vis, self.h_flux_vis):
            plot_widget.setXRange(visible_start, visible_end, padding=0.01)

    def update_graphs(self):
        """
        Update the graphs with new data
        """
        if self.index_plot_start == 0:
            self.index_plot_start += 1
            self.index_plot += 1
            # clear the graphs
            self.temperature_y = [np.nan] * len(self.temperature_x)
            self.temperature_y_target = [np.nan] * len(self.temperature_x)
            self.resistivity_y = [np.nan] * len(self.resistivity_x)
            self.h_flux_y = [np.nan] * len(self.h_flux_x)

        current_mode = tds_experiment.get_experiment_mode(self.config) == "CURRENT"
        plotted_target = np.nan if current_mode else self.target_temperature

        # Update the temperature graph
        if self.index_plot < len(self.temperature_y):
            self.temperature_y[self.index_plot] = self.temperature
            self.temperature_y_target[self.index_plot] = plotted_target
        else:
            self.temperature_x.append(self.temperature_x[-1] + 0.5)
            self.temperature_y.append(self.temperature)
            self.temperature_y_target.append(plotted_target)
        self.temperature_vis_line_target.setData(self.temperature_x, self.temperature_y_target)
        self.temperature_vis_line.setData(self.temperature_x, self.temperature_y)
        if current_mode:
            self.diff_label.setText(f"Power: {self.sample_power:.6f} W", color="#000000")
        else:
            self.diff_label.setText(
                f"Diff: {abs(self.target_temperature - self.temperature):.2f} C",
                color="#000000",
            )

        # Update the resistivity graph
        plotted_resistivity = self.resistivity if self._is_finite_number(self.resistivity) else np.nan
        if self.index_plot < len(self.resistivity_y):
            self.resistivity_y[self.index_plot] = plotted_resistivity
        else:
            self.resistivity_x.append(self.resistivity_x[-1] + 0.5)
            self.resistivity_y.append(plotted_resistivity)
        self.resistivity_vis_line.setData(self.resistivity_x, self.resistivity_y)

        # Update the heat flux graph
        if self.index_plot < len(self.h_flux_y):
            self.h_flux_y[self.index_plot] = 0.0
        else:
            self.h_flux_x.append(self.h_flux_x[-1] + 0.5)
            self.h_flux_y.append(0.0)
        self.h_flux_vis_line.setData(self.h_flux_x, self.h_flux_y)

        self.index_plot += 1
        self.refresh_plot_ranges()

    def _prepare_new_experiment_plots(self):
        """
        Clear plot traces for a new experiment start.
        """
        self.index_plot_start = 0
        self.index_plot = 0
        self.temperature_y = [np.nan] * len(self.temperature_x)
        self.temperature_y_target = [np.nan] * len(self.temperature_x)
        self.resistivity_y = [np.nan] * len(self.resistivity_x)
        self.h_flux_y = [np.nan] * len(self.h_flux_x)
        self.temperature_vis_line_target.setData(self.temperature_x, self.temperature_y_target)
        self.temperature_vis_line.setData(self.temperature_x, self.temperature_y)
        self.resistivity_vis_line.setData(self.resistivity_x, self.resistivity_y)
        self.h_flux_vis_line.setData(self.h_flux_x, self.h_flux_y)
        self.diff_label.setText("Diff: --", color="#000000")
        self.refresh_plot_ranges()

    def show_calibration_warning(self, message):
        """Display and retain a non-blocking T0 calibration warning."""
        self.t0_calibration_warning = str(message)
        self.error_message(f'WARNING: {message}', color='orange')

    def calibrate_base_temperature(self):
        """Anchor the loaded R-vs-T curve to the current wire at T0."""
        if not self.update_max_power():
            return
        if not self.require_loaded_curve('calibrating T. Zero'):
            return
        try:
            base_temperature = float(self.calib_temperature.text())
        except ValueError:
            self.error_message('Invalid base temperature', color='red')
            return
        if not self.update_calibration_start_current():
            return

        self.t0_calibration_warning = None
        self.emitter.last_calibration_warning = None
        self.emitter.reset_stop()
        self._set_operation_running(True)
        self.calibration_worker = CalibrationWorkerThread(
            calibration.calibrate_temperature_curve,
            self.emitter,
            self.r_vs_t,
            base_temperature,
            self.config,
        )
        self.calibration_worker.finished.connect(self.calibration_finished)
        self.calibration_worker.start()
        self.error_message('Running T. Zero calibration. Press Stop to cancel.', color='black')

    def calibration_finished(self, result):
        """
        Handles the result of the calibration worker thread.
        """
        self.calibration_worker = None
        self.emitter.reset_stop()
        self._set_operation_running(False)

        if isinstance(result, calibration.CalibrationCancelled):
            self.error_message('T. Zero calibration stopped.', color='black')
        elif isinstance(result, Exception):
            self.t_zero_calibrated = False
            self.error_message(f'Calibration failed: {result}', color='red')
        else:
            if result is not None:
                self.r_vs_t = result  # Update r_vs_t with the calibrated values
                self.t_zero_calibrated = True
                warning = self.t0_calibration_warning or self.emitter.last_calibration_warning
                if warning:
                    self.t0_calibration_warning = warning
                    self.error_message(f'Calibration successful. WARNING: {warning}', color='orange')
                else:
                    self.error_message('Calibration successful!', color='green')
            else:
                self.t_zero_calibrated = False
                self.error_message('Calibration failed and returned None', color='red')

    def pid_tuning_finished(self, result):
        """
        Handle the result of the guarded multi-point PID tuning worker.
        """
        controller_mode = tds_experiment.get_controller_mode(self.config)
        self.calibration_worker = None
        self.emitter.reset_stop()
        self._set_operation_running(False)

        if isinstance(result, calibration.CalibrationCancelled):
            print(f"{controller_mode} tuning stopped by user.")
            self.error_message(f'{controller_mode} tuning stopped.', color='black')
            return

        if isinstance(result, Exception):
            print(f"{controller_mode} tuning failed: {result}")
            self.error_message(f'{controller_mode} tuning failed: {result}', color='red')
            return

        schedule = result['schedule']
        self.config['pid_gain_schedule'] = schedule
        # Keep the flat pid_kp/ki/kd in sync too, as a fallback for any code
        # path that has not been updated to consult the schedule, and as the
        # value shown/edited when a schedule is later cleared.
        self.config['pid_kp'] = schedule[0]['kp']
        self.config['pid_ki'] = schedule[0]['ki']
        self.config['pid_kd'] = schedule[0]['kd']
        self.config['max_current_step_up'] = result['max_current_step_up']
        self.config['max_current_step_down'] = result['max_current_step_down']
        self.config['low_current_max_step_up'] = result['low_current_max_step_up']
        self.config['low_current_max_step_down'] = result['low_current_max_step_down']
        self.save_config()

        points_summary = ", ".join(
            f"{point['current_a']:.4f} A (Kp={point['kp']:.6f}, Ki={point['ki']:.6f}, Kd={point['kd']:.6f})"
            for point in schedule
        )
        print(
            f"{controller_mode} gain schedule tuned and saved at {len(schedule)} point(s): {points_summary}. "
            f"Step limits: low={result['low_current_max_step_up']:.4f} A, "
            f"normal={result['max_current_step_up']:.4f} A."
        )
        self.refresh_pid_status_label()
        self.error_message(
            f"{controller_mode} tuned at {len(schedule)} current(s); step limits suggested "
            f"(low={result['low_current_max_step_up']:.4f} A, normal={result['max_current_step_up']:.4f} A) "
            "- review before running. Save a Material Profile to reuse this on the same wire later.",
            color='black',
        )

    def calibrate_pid(self):
        """
        Calibrate the PID/PI gain schedule across the operating current range.
        """
        controller_mode = tds_experiment.get_controller_mode(self.config)
        if not self.update_max_power() or not self.update_calibration_start_current():
            return
        if not self.require_loaded_curve_and_t0(f'tuning {controller_mode}'):
            return

        try:
            self.experiment_params = self.parse_experiment_params()
            base_temperature = float(self.calib_temperature.text())
        except ValueError as exc:
            self.error_message(str(exc), color='red')
            return

        self.emitter.reset_stop()
        self._set_operation_running(True)
        self.calibration_worker = CalibrationWorkerThread(
            calibration.tune_pid_schedule,
            self.emitter,
            self.experiment_params[0],
            self.config,
            self.r_vs_t,
            base_temperature,
        )
        self.calibration_worker.finished.connect(self.pid_tuning_finished)
        self.calibration_worker.start()
        self.error_message(
            f'Running {controller_mode} tuning at multiple currents. Press Stop to cancel.', color='black'
        )

    def start_clicked(self):
        """
        Starts a new thread to execute the main functionality (replace with your logic).
        """
        if not self.update_max_power() or not self.update_calibration_start_current():
            return
        experiment_mode = tds_experiment.get_experiment_mode(self.config)
        if experiment_mode == "CURRENT":
            if not self.require_loaded_curve_and_t0('starting Current mode'):
                return
            try:
                self.experiment_params = self.parse_voltage_ramp_params()
                t_zero = None
            except ValueError as exc:
                self.error_message(str(exc), color='red')
                return
        else:
            if not self.require_loaded_curve_and_t0('starting the experiment'):
                return
            try:
                self.experiment_params = self.parse_experiment_params()
                t_zero = float(self.calib_temperature.text())
            except ValueError as exc:
                self.error_message(str(exc), color='red')
                return

        self.data_list = []
        self.experiment_name = self.sanitize_experiment_name(self.ex_name.text().strip() or 'TDS_test')
        self.current_experiment_dir = self.build_experiment_dir()
        try:
            results_curve, _, _ = tds_experiment._extend_curve_for_configured_extrapolation(
                self.r_vs_t,
                tds_experiment.build_control_config(self.config),
            )
            self.data_saver = ExperimentDataSaver(
                experiment_dir=self.current_experiment_dir,
                r_vs_t=results_curve,
                calibration_note=self.t0_calibration_warning,
                flush_interval_s=self.config['autosave_flush_interval_s'],
                batch_size=self.config['autosave_batch_size'],
                run_metadata=self.build_run_metadata(),
            ).start()
        except Exception as exc:
            self.data_saver = None
            self.current_experiment_dir = None
            self.error_message(f'Cannot start autosave: {exc}', color='red')
            return
        self.emitter.reset_stop()
        self._prepare_new_experiment_plots()
        self._set_operation_running(True)

        if experiment_mode == "CURRENT":
            self.worker_thread = WorkerThread(
                tds_experiment.current_ramp,
                emitter=self.emitter,
                ramp_params=self.experiment_params,
                r_vs_t=self.r_vs_t,
                config=self.config,
                data_saver=self.data_saver,
            )
        else:
            self.worker_thread = WorkerThread(
                tds_experiment.tds,
                emitter=self.emitter,
                experiment_params=self.experiment_params,
                r_vs_t=self.r_vs_t,
                config=self.config,
                t_zero=t_zero,
                data_saver=self.data_saver,
            )
        self.worker_thread.finished.connect(self.thread_finished)
        self.worker_thread.start()
        self.update_timer.start(500)
        if experiment_mode == "CURRENT":
            self.error_message(f'Voltage ramp started. Autosaving to {self.current_experiment_dir}', color='black')
        else:
            self.error_message(f'Experiment started. Autosaving to {self.current_experiment_dir}', color='black')

    def stop_clicked(self):
        """
        Sends a stop signal to the thread and disables the Stop button.
        """
        if (
            (self.worker_thread is not None and self.worker_thread.isRunning())
            or (self.calibration_worker is not None and self.calibration_worker.isRunning())
        ):
            self.emitter.emit_stop()
            self.stop_botton.setEnabled(False)
            self.error_message('Stopping current operation...', color='black')
            print('Stop signal sent')

    def thread_finished(self, finished):
        """
        Handles the thread completion.
        """
        self.update_timer.stop()
        self.voltage = 0
        self.current = 0
        self.temperature = 0
        self.resistivity = 0
        self.commanded_current = 0
        self.sample_power = 0
        self._set_operation_running(False)
        self.emitter.reset_stop()
        self.worker_thread = None

        self.voltage_lcd.display(0)
        self.current_lcd.display(0)
        self.resistivity_lcd.display(0)
        self.power_lcd.display(0)
        self.temperature_lcd.display(0)
        self.temperature_target_lcd.display(0)
        # After stop/finish show full experiment history and keep it visible.
        if self.plot_window_button.isChecked():
            self.plot_window_button.setChecked(False)
        else:
            self.plot_window_last60_selected = False
            self.refresh_plot_ranges()

        self.experiment_params = []
        experiment_dir = self.current_experiment_dir
        self.current_experiment_dir = None
        self.data_saver = None
        # Save new value of experiment counter
        if EXPERIMENT_COUNTER_PATH.exists():
            self.ex_counter += 1
            with EXPERIMENT_COUNTER_PATH.open('w') as f:
                f.write(str(self.ex_counter))
        self.ex_number.setText(str(self.ex_counter))

        if finished is not None:
            self.error_message(f'Experiment stopped with error: {finished}', color='red')
            print(f"Error in experiment thread: {finished}")
        elif not self.data_list:
            self.error_message('Experiment finished without recorded data', color='red')
        else:
            self.error_message(f'Experiment finished. Data saved in {experiment_dir}', color='green')
        self.data_list = []

    def error_message(self, message, color='red'):
        """
        Display an error message and start a timer to hide it after 8 seconds

        Args:
            message (str): Error message to display

        Return:
            None
        """
        color_map = {
            'red': '#ff0000',
            'green': '#008000',
            'orange': '#c45f00',
            'black': '#000000',
        }
        html_color = color_map.get(color, '#000000')
        _translate = QtCore.QCoreApplication.translate
        self.Error.setText(_translate("OXCART",
                                      "<html><head/><body><p><span style=\" color:"
                                      + html_color + ";\">"
                                      + message + "</span></p></body></html>"))

        self.timer_error.start(8000)

    def hideMessage(self, ):
        """
        Hide the message and stop the timer
        Args:
            None

        Return:
            None
        """
        # Hide the message and stop the timer
        _translate = QtCore.QCoreApplication.translate
        self.Error.setText(_translate("OXCART",
                                      "<html><head/><body><p><span style=\" "
                                      "color:#ff0000;\"></span></p></body></html>"))

        self.timer_error.stop()


class WorkerThread(QThread):
    finished = pyqtSignal(object)  # Signal emitted when the function is done

    def __init__(self, func, emitter, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.stop_flag = False
        self.emitter = emitter  # Keep reference to the emitter

        # Connect the stop signal to the stop flag
        emitter.stop_signal.connect(self.set_stop_flag)

    def set_stop_flag(self):
        """
        Set the stop flag to True when the stop signal is emitted.
        """
        self.stop_flag = True

    def run(self):
        try:
            # Execute the function with the stop flag passed as an argument
            self.func(self.emitter, *self.args, **self.kwargs)
            self.finished.emit(None)  # Emit when finished without exception
        except Exception as e:
            traceback.print_exc()
            self.finished.emit(e)  # Emit the exception if any error occurs

class CalibrationWorkerThread(QThread):
    finished = pyqtSignal(object)  # Signal emitted when the function is done (can be result or error)

    def __init__(self, func, emitter, *args, **kwargs):
        super().__init__()
        self.func = func
        self.emitter = emitter
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            # Execute the function and get the return value
            result = self.func(*self.args, emitter=self.emitter, **self.kwargs)
            self.finished.emit(result)  # Emit the result when finished successfully
        except Exception as e:
            traceback.print_exc()
            self.finished.emit(e)  # Emit the exception if any error occurs


class SignalEmitter(QtCore.QObject):
    stop_signal = pyqtSignal()  # Signal to stop the thread
    experiment_signal = pyqtSignal(list)  # Signal to emit experiment data
    live_measurement_signal = pyqtSignal(object)  # Signal for calibration/tuning live readings
    calibration_warning_signal = pyqtSignal(str)  # Non-blocking T0 mismatch warning
    max_power_signal = pyqtSignal(float)  # Signal to emit the maximum measured sample power
    max_current_signal = pyqtSignal(float)  # Signal to emit the maximum current

    def __init__(self):
        super().__init__()
        self.stopped = False
        self.last_calibration_warning = None

    def emit_stop(self):
        self.stopped = True
        self.stop_signal.emit()

    def reset_stop(self):
        self.stopped = False


class TDSMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = None

    def closeEvent(self, event):
        if self.ui is not None and not self.ui.can_close_window():
            event.ignore()
            return
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    TDS = TDSMainWindow()
    try:
        ensure_runtime_dirs()
        data = config_io.load_config()
    except Exception as e:
        print('Cannot load the configuration file')
        print(e)
        return 1

    ui = Ui_TDS(data)
    ui.setupUi(TDS)
    TDS.ui = ui
    TDS.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())


