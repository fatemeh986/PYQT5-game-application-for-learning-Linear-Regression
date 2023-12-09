import numpy as np
import warnings
import math

warnings.filterwarnings("ignore")
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QMainWindow, QApplication, QSizePolicy
from PyQt5.QtGui import QPen, QBrush
import pyqtgraph as pg
import numpy as np
from projectDesign import Ui_MainWindow


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi()
        self.initialize_variables()
        self.setupPlots()
        self.ui.noiseType.currentIndexChanged.connect(self.add_noise_to_signal)
        self.ui.binsSlider.valueChanged.connect(self.plot_noise_distribution)
        self.ui.aSlider.valueChanged.connect(self.generate_signal)
        self.ui.bSlider.valueChanged.connect(self.generate_signal)
        self.ui.nSlider.valueChanged.connect(self.generate_signal)
        self.ui.meanSlider.valueChanged.connect(self.add_noise_to_signal)
        self.ui.sigmaSlider.valueChanged.connect(self.add_noise_to_signal)

    def setupPlots(self):
        self.initial_signal_plot_widget = self.createPlotWidget()
        self.noisy_signal_plot_widget = self.createPlotWidget()
        self.noise_plot_widget = self.createPlotWidget()
        self.noise_distribution_plot_widget = self.createPlotWidget()
        self.ui.init_signal_vlayout.addWidget(
            self.initial_signal_plot_widget, stretch=1
        )
        self.ui.noisy_signal_vlayout.addWidget(self.noisy_signal_plot_widget, stretch=1)
        self.ui.noise_distribution_vlayout.addWidget(
            self.noise_distribution_plot_widget, stretch=1
        )
        self.ui.noisePlotVLayout.addWidget(self.noise_plot_widget, stretch=1)

    def createPlotWidget(self):
        plot_widget = pg.PlotWidget()
        plot_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        return plot_widget

    def initialize_variables(self):
        self.initial_signal = []
        self.noisy_signal = []
        self.noise = []
        self.estimated_a = 0
        self.estimated_b = 0
        self.estimated_sigma = 0
        self.a_error = 0
        self.b_error = 0
        self.sigma_error = 0

    def connect_sliders(self):
        sliders = [
            self.ui.aSlider,
            self.ui.bSlider,
            self.ui.nSlider,
            self.ui.meanSlider,
            self.ui.sigmaSlider,
        ]
        for slider in sliders:
            slider.valueChanged.connect(self.process_signal)

    def setupUi(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

    def generate_signal(self):
        a = int(self.ui.aSlider.value())
        b = int(self.ui.bSlider.value())
        n = int(self.ui.nSlider.value())

        self.ui.initialALabel.setText(str(a))
        self.ui.initialBLabel.setText(str(b))
        self.ui.initialSigmaLabel.setText(str(self.ui.sigmaSlider.value()))

        self.initial_signal = [a * x + b for x in range(n)]

        self.plot_initial_signal()
        self.add_noise_to_signal()

    def plot_initial_signal(self):
        self.initial_signal_plot_widget.clear()

        self.initial_signal_plot_widget.plot(
            range(len(self.initial_signal)), self.initial_signal
        )

        self.initial_signal_plot_widget.setTitle("Initial Signal")
        self.initial_signal_plot_widget.setLabel("left", "Y")
        self.initial_signal_plot_widget.setLabel("bottom", "X")

    def add_noise_to_signal(self):
        noise_type = self.ui.noiseType.currentText()
        sigma = int(self.ui.sigmaSlider.value())
        mean = int(self.ui.meanSlider.value())

        if noise_type == "Gaussian":
            print("gaussian")
            self.noise = np.random.normal(sigma, mean, size=len(self.initial_signal))
            self.noisy_signal = self.initial_signal + self.noise

        else:
            print("uniform")
            low = mean - sigma * math.sqrt(3)
            high = mean + sigma * math.sqrt(3)
            self.noise = np.random.uniform(low, high, size=len(self.initial_signal))
            self.noisy_signal = self.initial_signal + self.noise

        self.find_linear_trend()
        self.plot_noisy_signal()

    def plot_noisy_signal(self):
        self.noisy_signal_plot_widget.clear()
        x_values = np.arange(self.ui.nSlider.value())
        projection = self.estimated_a * x_values + self.estimated_b
        self.noisy_signal_plot_widget.plot(
            range(len(self.noisy_signal)), self.noisy_signal
        )
        self.noisy_signal_plot_widget.plot(range(len(self.noisy_signal)), projection)

        self.noisy_signal_plot_widget.setTitle("Noisy Signal")
        self.noisy_signal_plot_widget.setLabel("left", "Y")
        self.noisy_signal_plot_widget.setLabel("bottom", "X")

    def find_linear_trend(self):
        if len(self.noisy_signal) > 1:
            x_indices = np.arange(len(self.noisy_signal))
            self.estimated_a, self.estimated_b = np.polyfit(
                x_indices, self.noisy_signal, 1
            )  # 1 indicates linear fit
            self.calculate_noise()

    def calculate_noise(self):
        if len(self.noisy_signal) > 1:
            x_values = np.arange(self.ui.nSlider.value())
            trend = self.estimated_a * x_values + self.estimated_b
            residuals = self.noisy_signal - trend
            self.estimated_sigma = np.std(residuals)

            self.ui.estimatedALabel.setText(str(self.estimated_a))
            self.ui.estimatedBLabel.setText(str(self.estimated_b))
            self.ui.estimatedSigmaLabel.setText(str(self.estimated_sigma))
            self.calculate_error()
            self.plot_noise()
            self.plot_noise_distribution()

    def plot_noise(self):
        self.noise_plot_widget.clear()

        self.noise_plot_widget.plot(range(len(self.initial_signal)), self.noise)

        self.noise_plot_widget.setTitle("Noise")
        self.noise_plot_widget.setLabel("left", "Y")
        self.noise_plot_widget.setLabel("bottom", "X")

    def plot_noise_distribution(self):
        self.noise_distribution_plot_widget.clear()

        hist, edges = np.histogram(self.noise, bins=int(self.ui.binsSlider.value()))

        histogram_plot_item = pg.PlotDataItem(
            x=edges,
            y=hist,
            stepMode=True,
            fillLevel=0,
            brush=QBrush(Qt.blue),  # Customize the bar color as needed
            pen=QPen(Qt.black),
        )

        pen = pg.mkPen(width=1)  # Customize the line color and width as needed
        histogram_plot_item.setPen(pen)

        # Add the PlotDataItem to the PlotWidget
        self.noise_distribution_plot_widget.addItem(histogram_plot_item)

        # self.noise_distribution_plot_widget.plot(range(len(self.initial_signal)), self.noise)

        self.noise_distribution_plot_widget.setTitle("Noise")
        self.noise_distribution_plot_widget.setLabel("left", "Y")
        self.noise_distribution_plot_widget.setLabel("bottom", "X")

    def calculate_error(self):
        if self.estimated_a != 0 and self.estimated_b != 0:
            self.a_error = (
                np.abs(self.estimated_a - int(self.ui.aSlider.value()))
                / self.estimated_a
            )
            self.b_error = (
                np.abs(self.estimated_b - int(self.ui.bSlider.value()))
                / self.estimated_b
            )
            self.sigma_error = (
                np.abs(self.estimated_sigma - int(self.ui.sigmaSlider.value()))
                / self.estimated_sigma
            )
            a = self.ui.aSlider.value()
            b = self.ui.bSlider.value()
            sigma = self.ui.sigmaSlider.value()
            self.ui.errorALabel.setText(str(self.a_error))
            self.ui.errorBLabel.setText(str(self.b_error))
            self.ui.errorSigmaLabel.setText(str(self.sigma_error))

            observed_values = np.array([a, b, sigma])
            predicted_values = np.array(
                [
                    self.estimated_a,
                    self.estimated_b,
                    self.estimated_sigma,
                ]
            )
            errors = observed_values - predicted_values

            mse = np.mean(errors**2)

            mean_observed = np.mean(observed_values)
            squared_differences = (observed_values - mean_observed) ** 2
            tss = np.sum(squared_differences)

            r_squared = 1 - (mse / tss)

            self.ui.RLabel.setText(f"R^2={r_squared}")


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())
