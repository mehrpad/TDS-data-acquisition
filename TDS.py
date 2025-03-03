import json

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
import pandas as pd
import os
import tds_experiment
import h5py
import pyqtgraph as pg

import calibration


class Ui_TDS(object):
    def __init__(self, data):
        self.target_temperature = 0
        self.index_plot = 0
        self.voltage = 0
        self.current = 0
        self.temperature = 0
        self.index_plot_start = 0
        self.config = data
        self.experiment_params = []
        self.r_vs_t = None
        self.data_list = []
        self.worker_thread = None
        self.file_path = None
        self.experiment_name = 'TDS_test'
        if os.path.exists("./files/experiment_counter.txt"):
            # Read the experiment counter
            with open('./files/experiment_counter.txt') as f:
                self.ex_counter = int(f.readlines()[0])
        else:
            # create a new txt file
            with open('./files/experiment_counter.txt', 'w') as f:
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
        ]

    def setupUi(self, TDS):
        TDS.setObjectName("TDS")
        TDS.resize(820, 465)
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
        self.max_voltage = QtWidgets.QLineEdit(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.max_voltage.sizePolicy().hasHeightForWidth())
        self.max_voltage.setSizePolicy(sizePolicy)
        self.max_voltage.setMinimumSize(QtCore.QSize(100, 20))
        self.max_voltage.setStyleSheet("QLineEdit{\n"
                                       "                                                background: rgb(223,223,233)\n"
                                       "                                                }\n"
                                       "                                            ")
        self.max_voltage.setObjectName("max_voltage")
        self.gridLayout.addWidget(self.max_voltage, 3, 1, 1, 1)
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
        self.temperature_vis.setMinimumSize(QtCore.QSize(150, 150))
        self.temperature_vis.setStyleSheet("QWidget{\n"
                                           "                                            border: 0.5px solid gray;\n"
                                           "                                            }\n"
                                           "                                        ")
        self.temperature_vis.setObjectName("temperature_vis")
        self.gridLayout_4.addWidget(self.temperature_vis, 0, 0, 1, 1)
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
        self.h_flux_vis.setMinimumSize(QtCore.QSize(150, 150))
        self.h_flux_vis.setStyleSheet("QWidget{\n"
                                      "                                            border: 0.5px solid gray;\n"
                                      "                                            }\n"
                                      "                                        ")
        self.h_flux_vis.setObjectName("h_flux_vis")
        self.gridLayout_4.addWidget(self.h_flux_vis, 1, 0, 1, 1)
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
        self.gridLayout_3.addWidget(self.parameters_text, 2, 0, 1, 4)
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
        self.gridLayout_3.addLayout(self.gridLayout_2, 3, 0, 1, 3)
        self.gridLayout_5.addLayout(self.gridLayout_3, 1, 0, 1, 1)
        self.Error = QtWidgets.QLabel(parent=self.centralwidget)
        self.Error.setMinimumSize(QtCore.QSize(800, 30))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setStrikeOut(False)
        self.Error.setFont(font)
        self.Error.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.Error.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse)
        self.Error.setObjectName("Error")
        self.gridLayout_5.addWidget(self.Error, 2, 0, 1, 2)
        self.gridLayout_6.addLayout(self.gridLayout_5, 0, 0, 1, 1)
        TDS.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=TDS)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 820, 22))
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
        TDS.setTabOrder(self.calib_temperature, self.max_voltage)
        TDS.setTabOrder(self.max_voltage, self.max_current)
        TDS.setTabOrder(self.max_current, self.parameters_text)
        TDS.setTabOrder(self.parameters_text, self.start_botton)
        TDS.setTabOrder(self.start_botton, self.stop_botton)
        TDS.setTabOrder(self.stop_botton, self.find_csv_botton)
        TDS.setTabOrder(self.find_csv_botton, self.load_csv_botton)
        TDS.setTabOrder(self.load_csv_botton, self.calibrate_botton_base_t)
        TDS.setTabOrder(self.calibrate_botton_base_t, self.calibrate_botton_pid)
        TDS.setTabOrder(self.calibrate_botton_pid, self.temperature_vis)
        TDS.setTabOrder(self.temperature_vis, self.h_flux_vis)

        #####
        self.timer_error = QtCore.QTimer()
        self.timer_error.timeout.connect(self.hideMessage)

        self.emitter.experiment_signal.connect(self.update_experiment_signal)

        self.find_csv_botton.clicked.connect(self.find_csv_clicked)
        self.load_csv_botton.clicked.connect(self.load_csv_clicked)
        self.start_botton.clicked.connect(self.start_clicked)
        self.stop_botton.clicked.connect(self.stop_clicked)
        self.calibrate_botton_base_t.clicked.connect(self.calibrate_base_temperature)
        self.calibrate_botton_pid.clicked.connect(self.calibrate_pid)
        self.calibrate_botton_pid.setEnabled(False)

        self.h_flux_x = [i * 0.5 for i in range(200)]
        self.h_flux_y = [0.0] * 200
        self.h_flux_y = [np.nan] * len(self.h_flux_x)
        pen_h_flux = pg.mkPen(color=(0, 0, 225), width=4)
        self.h_flux_vis_line = self.h_flux_vis.plot(self.h_flux_x, self.h_flux_y, pen=pen_h_flux)

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
                                                       f"{abs(self.target_temperature - self.temperature):.2f}°C")

        # Add Axis Labels
        self.styles = {"color": "#f00", "font-size": "12px"}
        self.temperature_vis.setLabel("left", "Temperature", units='°C', **self.styles)
        self.temperature_vis.setLabel("bottom", "Time (s)", **self.styles)
        self.h_flux_vis.setLabel("left", "Flux", units='mol/s', **self.styles)
        self.h_flux_vis.setLabel("bottom", "Time (s)", **self.styles)

        self.temperature_vis.showGrid(x=True, y=True)
        self.h_flux_vis.showGrid(x=True, y=True)

        self.update_timer = QTimer()  # Create a QTimer for updating graphs
        # self.update_timer.start(500)
        self.update_timer.timeout.connect(self.update_graphs)  # Connect it to the update_graphs slot

        self.ex_number.setEnabled(False)
        self.ex_number.setText(str(self.ex_counter))

        self.max_voltage.editingFinished.connect(self.update_max_voltage)
        self.max_current.editingFinished.connect(self.update_max_current)

        # from config file put max voltage and current
        self.max_voltage.setText(str(self.config['max_voltage']))
        self.max_current.setText(str(self.config['max_current']))

    def retranslateUi(self, TDS):
        _translate = QtCore.QCoreApplication.translate
        TDS.setWindowTitle(_translate("TDS", "TDS"))
        self.label_183.setText(_translate("TDS", "Experiment Number"))
        self.ex_number.setText(_translate("TDS", "1"))
        self.label_175.setText(_translate("TDS", "Experiment Name"))
        self.ex_name.setText(_translate("TDS", "test"))
        self.label_176.setText(_translate("TDS", "Zero Temperature (°C)"))
        self.calib_temperature.setText(_translate("TDS", "23"))
        self.label_177.setText(_translate("TDS", "Max Voltage (V)"))
        self.max_voltage.setText(_translate("TDS", "10"))
        self.label_178.setText(_translate("TDS", "Max Current (A)"))
        self.max_current.setText(_translate("TDS", "5"))
        self.label_4.setText(_translate("TDS", "Target Temp. (°C)    "))
        self.label_1.setText(_translate("TDS", "Measured Temp (°C)"))
        self.label_2.setText(_translate("TDS", "Voltage (V)              "))
        self.label_3.setText(_translate("TDS", "Current (A)                "))
        self.parameters_text.setHtml(_translate("TDS",
                                                "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                                "<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
                                                "p, li { white-space: pre-wrap; }\n"
                                                "</style></head><body style=\" font-family:\'Segoe UI\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
                                                "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'JetBrains Mono,monospace\'; font-size:8pt; color:#000000;\">{start_T=200;step_T</span><span style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.875pt;\">=40;target_T=800;ramp_speed_c_min=20;hold_step_time_min=1</span><span style=\" font-family:\'JetBrains Mono,monospace\'; font-size:8pt; color:#000000;\">}</span>                                                                              </p></body></html>"))
        self.start_botton.setText(_translate("TDS", "Start"))
        self.find_csv_botton.setText(_translate("TDS", "Find R vs. T"))
        self.calibrate_botton_base_t.setText(_translate("TDS", "Calibrate T. Zero"))
        self.stop_botton.setText(_translate("TDS", "Stop"))
        self.load_csv_botton.setText(_translate("TDS", "Load"))
        self.calibrate_botton_pid.setText(_translate("TDS", "Tune PID"))
        self.Error.setText(_translate("TDS", "<html><head/><body><p><br/></p></body></html>"))
        self.menuFile.setTitle(_translate("TDS", "File"))
        self.menuHelp.setTitle(_translate("TDS", "Help"))
        self.actionExit.setText(_translate("TDS", "Exit"))

    def update_max_voltage(self):
        """
        Update the maximum voltage
        """
        self.config['max_voltage'] = float(self.max_voltage.text())
        self.emitter.max_voltage_signal.emit(self.config['max_voltage'])
    def update_max_current(self):
        """
        Update the maximum current
        """
        self.config['max_current'] = float(self.max_current.text())
        self.emitter.max_current_signal.emit(self.config['max_current'])
    def find_csv_clicked(self):
        """
        Opens a file dialog in a separate thread to avoid blocking the UI.
        """
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select File", "", "Excel Files (*.xlsx);;CSV Files (*.csv)"
        )
        if file_path:
            self.file_path = file_path
            self.error_message(f"Selected file: {file_path}", color='black')

    def load_csv_clicked(self):
        """
        Loads the selected CSV file in a separate thread to avoid blocking the UI.
        """
        if self.file_path.endswith(".csv") or self.file_path.endswith(".xlsx"):
            if self.file_path.endswith(".csv"):
                df = pd.read_csv(self.file_path)
            elif self.file_path.endswith(".xlsx"):
                df = pd.read_excel(self.file_path, header=1)
            self.error_message('File loaded', color='black')
            resistivity = df['resistivity'].to_numpy()
            temperature = df['temperature [C]'].to_numpy()
            self.r_vs_t = np.vstack((resistivity, temperature))
        else:
            raise ValueError("Invalid file type")

    def update_experiment_signal(self, data):
        """
        Updates the experiment signal with the new data

        Args:
            data (list): List of data to update the experiment signal

        Return:
            None
        """
        # print(f"Received data: time: {data[0]}, set_T: {data[1]}, T: {data[2]}, h_f: {data[3]}, V: {data[4]}, "
        #       f"I: {data[5]}, C_V: {data[6]}")
        self.target_temperature = data[1]
        self.voltage = data[4]
        self.current = data[5]
        self.temperature = data[2]
        self.voltage_lcd.display(self.voltage)
        self.current_lcd.display(self.current)
        self.temperature_lcd.display(self.temperature)
        self.temperature_target_lcd.display(self.target_temperature)

        # "time",  # Time in UNIX-readable format
        # "set_T",  # Set temperature
        # "T",  # Measured temperature
        # "h_f"  # Heat flux
        # "V",  # Voltage
        # "I",  # Current
        # "C_V",  # Calculated Power supply voltage
        self.data_list.append([data[0], data[1], data[2], data[3], data[4], data[5], data[6]])

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
            self.h_flux_y = [np.nan] * len(self.h_flux_x)

        # Update the temperature graph
        if self.index_plot < len(self.temperature_y):
            self.temperature_y[self.index_plot] = self.temperature
            self.temperature_y_target[self.index_plot] = self.target_temperature
        else:
            self.temperature_x.append(self.temperature_x[-1] + 0.5)
            self.temperature_y.append(self.temperature)
            self.temperature_y_target.append(self.target_temperature)
        self.temperature_vis_line_target.setData(self.temperature_x, self.temperature_y_target)
        self.temperature_vis_line.setData(self.temperature_x, self.temperature_y)
        self.legend.items[0][1].setText(f"Diff: {abs(self.target_temperature - self.temperature):.2f}°C")

        # Update the heat flux graph
        if self.index_plot < len(self.h_flux_y):
            self.h_flux_y[self.index_plot] = 0.0
        else:
            self.h_flux_x.append(self.h_flux_x[-1] + 0.5)
            self.h_flux_y.append(0.0)
        self.h_flux_vis_line.setData(self.h_flux_x, self.h_flux_y)

        self.index_plot += 1

    def calibrate_base_temperature(self):
        """
        Calibrate the base temperature
        """
        if self.r_vs_t is not None:
            self.calibrate_botton_base_t.setEnabled(False)
            self.calibrate_botton_pid.setEnabled(False)
            self.find_csv_botton.setEnabled(False)
            self.load_csv_botton.setEnabled(False)
            self.start_botton.setEnabled(False)
            self.stop_botton.setEnabled(False)
            try:
                base_temperature = float(self.calib_temperature.text())
                self.calibration_worker = CalibrationWorkerThread(
                    calibration.calibrate_temperature_curve, self.r_vs_t, base_temperature
                )
                self.calibration_worker.finished.connect(self.calibration_finished)
                self.calibration_worker.start()
            except ValueError:
                self.error_message('Invalid base temperature', color='red')
                self.calibrate_botton_base_t.setEnabled(True)
                self.calibrate_botton_pid.setEnabled(True)
                self.find_csv_botton.setEnabled(True)
                self.load_csv_botton.setEnabled(True)
                self.start_botton.setEnabled(True)
                self.stop_botton.setEnabled(True)
        else:
            self.error_message('First Load a R vs. T', color='red')

    def calibration_finished(self, result):
        """
        Handle the completion of the calibration process
        """
        if result is None:
            self.error_message('Zero temperature calibrated', color='black')
        else:
            self.error_message('Error calibrating temperature', color='red')
        self.calibrate_botton_base_t.setEnabled(True)
        self.calibrate_botton_pid.setEnabled(True)
        self.find_csv_botton.setEnabled(True)
        self.load_csv_botton.setEnabled(True)
        self.start_botton.setEnabled(True)
        self.stop_botton.setEnabled(True)

    def calibrate_pid(self):
        """
        Calibrate the PID
        """
        self.calibrate_botton_base_t.setEnabled(False)
        self.calibrate_botton_pid.setEnabled(False)
        self.find_csv_botton.setEnabled(False)
        self.load_csv_botton.setEnabled(False)
        self.start_botton.setEnabled(False)
        self.stop_botton.setEnabled(False)
        # Extract each line and parse key-value pairs
        text = self.parameters_text.toPlainText()
        for line in text.strip().split("\n"):
            line = line.strip("{}").strip()  # Remove surrounding braces and extra spaces
            # Split by semicolons and process key-value pairs
            key_value_pairs = line.split(";")
            single_dict = {key.strip(): float(value.strip(" }")) for key, value in
                           (pair.split("=") for pair in key_value_pairs)}
            self.experiment_params.append(single_dict)

        try:
            calibration.tune_pid(self.experiment_params[0], self.config, self.r_vs_t)
            self.error_message('PID calibrated', color='black')
        except ValueError:
            self.error_message('Error calibrating PID', color='red')
        self.calibrate_botton_base_t.setEnabled(True)
        self.calibrate_botton_pid.setEnabled(True)
        self.find_csv_botton.setEnabled(True)
        self.load_csv_botton.setEnabled(True)
        self.start_botton.setEnabled(True)
        self.stop_botton.setEnabled(True)

    def start_clicked(self):
        """
        Starts a new thread to execute the main functionality (replace with your logic).
        """
        self.start_botton.setEnabled(False)
        self.stop_botton.setEnabled(True)
        self.calibrate_botton_base_t.setEnabled(False)
        self.find_csv_botton.setEnabled(False)
        self.load_csv_botton.setEnabled(False)
        self.ex_name.setEnabled(False)

        # Extract each line and parse key-value pairs
        text = self.parameters_text.toPlainText()
        for line in text.strip().split("\n"):
            line = line.strip("{}").strip()  # Remove surrounding braces and extra spaces
            # Split by semicolons and process key-value pairs
            key_value_pairs = line.split(";")
            single_dict = {key.strip(): float(value.strip(" }")) for key, value in
                           (pair.split("=") for pair in key_value_pairs)}
            self.experiment_params.append(single_dict)

        self.worker_thread = WorkerThread(tds_experiment.tds, emitter=self.emitter,
                                          experiment_params=self.experiment_params, r_vs_t=self.r_vs_t,
                                          config=self.config, t_zero=float(self.calib_temperature.text()))
        self.worker_thread.finished.connect(self.thread_finished)
        self.worker_thread.start()
        self.update_timer.start(500)
        # tds_experiment.tds(self.emitter, self.experiment_params, self.r_vs_t, self.config)

    def stop_clicked(self):
        """
        Sends a stop signal to the thread and disables the Stop button.
        """
        if self.worker_thread is not None:
            self.emitter.emit_stop()
            self.stop_botton.setEnabled(False)
            print('Stop signal sent')

    def thread_finished(self, finished):
        """
        Handles the thread completion.
        """
        self.update_timer.stop()
        self.index_plot_start = 0
        self.index_plot = 0
        self.voltage = 0
        self.current = 0
        self.temperature = 0
        self.start_botton.setEnabled(True)
        self.stop_botton.setEnabled(True)
        self.calibrate_botton_base_t.setEnabled(True)
        self.find_csv_botton.setEnabled(True)
        self.load_csv_botton.setEnabled(True)
        self.ex_name.setEnabled(True)

        self.emitter.stopped = False
        self.worker_thread = None

        self.voltage_lcd.display(0)
        self.current_lcd.display(0)
        self.temperature_lcd.display(0)
        self.temperature_target_lcd.display(0)

        self.experiment_params = []
        # Save new value of experiment counter
        if os.path.exists("./files/experiment_counter.txt"):
            self.ex_counter += 1
            with open('./files/experiment_counter.txt', 'w') as f:
                f.write(str(self.ex_counter))
        self.ex_number.setText(str(self.ex_counter))

        if finished is not None:
            self.error_message(f"Error in experiment thread")
            print(f"Error in experiment thread: {finished}")
        else:
            # dataset = pd.DataFrame(self.data_list, columns=self.columns)
            # print(dataset)
            dataset = np.array(self.data_list)
            dataset = dataset.T
            # create a new folder for the experiment
            path = f"./data/{self.ex_counter}_{self.experiment_name}"
            if not os.path.exists(path):
                os.makedirs(path, mode=0o777, exist_ok=True)
            with h5py.File(path + '/data.h5', "w") as f:
                f.create_dataset("time_stamp", data=dataset[0], dtype='f8')
                f.create_dataset("set_temperature", data=dataset[1], dtype='f8')
                f.create_dataset("measured_temperature", data=dataset[2], dtype='f8')
                f.create_dataset("heat_flux", data=dataset[3], dtype='f8')
                f.create_dataset("voltage", data=dataset[4], dtype='f8')
                f.create_dataset("current", data=dataset[5], dtype='f8')
                f.create_dataset("calculated_voltage", data=dataset[6], dtype='f8')

    def error_message(self, message, color='red'):
        """
        Display an error message and start a timer to hide it after 8 seconds

        Args:
            message (str): Error message to display

        Return:
            None
        """
        if color == 'red':
            _translate = QtCore.QCoreApplication.translate
            self.Error.setText(_translate("OXCART",
                                          "<html><head/><body><p><span style=\" color:#ff0000;\">"
                                          + message + "</span></p></body></html>"))
        else:
            _translate = QtCore.QCoreApplication.translate
            self.Error.setText(_translate("OXCART",
                                          "<html><head/><body><p><span style=\" color:#000000;\">"
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
            self.finished.emit(e)  # Emit the exception if any error occurs

class CalibrationWorkerThread(QThread):
    finished = pyqtSignal(object)  # Signal emitted when the function is done

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            # Execute the function
            self.func(*self.args, **self.kwargs)
            self.finished.emit(None)  # Emit when finished without exception
        except Exception as e:
            self.finished.emit(e)  # Emit the exception if any error occurs


class SignalEmitter(QtCore.QObject):
    stop_signal = pyqtSignal()  # Signal to stop the thread
    experiment_signal = pyqtSignal(list)  # Signal to emit experiment data
    max_voltage_signal = pyqtSignal(float)  # Signal to emit the maximum voltage
    max_current_signal = pyqtSignal(float)  # Signal to emit the maximum current

    def __init__(self):
        super().__init__()
        self.stopped = False

    def emit_stop(self):
        self.stopped = True
        self.stop_signal.emit()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    TDS = QtWidgets.QMainWindow()
    try:
        # Load the JSON file
        config_file = './files/config.json'
        with open(config_file) as file:
            data = json.load(file)
    except Exception as e:
        print('Cannot load the configuration file')
        print(e)
        sys.exit()

    ui = Ui_TDS(data)
    ui.setupUi(TDS)
    TDS.show()
    sys.exit(app.exec())


