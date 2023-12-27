import warnings
import math

warnings.filterwarnings("ignore")
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QMainWindow, QApplication, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
from projectDesign import Ui_MainWindow
from scipy.stats import norm, uniform, linregress
from sklearn.metrics import r2_score


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi()
        self.initialize_variables()
        self.setupPlots()
        self.setWindowTitle("Linear Regression Estimator")
        self.ui.noiseType.currentIndexChanged.connect(self.add_noise_to_signal)
        self.ui.binsSlider.valueChanged.connect(self.plot_noise_distribution)
        self.ui.aSlider.valueChanged.connect(self.generate_signal)
        self.ui.bSlider.valueChanged.connect(self.generate_signal)
        self.ui.nSlider.valueChanged.connect(self.generate_signal)
        self.ui.meanSlider.valueChanged.connect(self.add_noise_to_signal)
        self.ui.sigmaSlider.valueChanged.connect(self.add_noise_to_signal)

    def setupPlots(self):
        self.initial_signal_plot_widget = FigureCanvas(plt.Figure())
        self.noisy_signal_plot_widget = FigureCanvas(plt.Figure())
        self.noise_plot_widget = FigureCanvas(plt.Figure())
        self.noise_distribution_plot_widget = FigureCanvas(plt.Figure())
        self.ui.init_signal_vlayout.addWidget(
            self.initial_signal_plot_widget, stretch=1
        )
        self.ui.noisy_signal_vlayout.addWidget(
            self.noise_distribution_plot_widget, stretch=1
        )  # noisy_signal_plot_widget
        self.ui.noise_distribution_vlayout.addWidget(
            self.noisy_signal_plot_widget, stretch=1
        )
        self.ui.noisePlotVLayout.addWidget(self.noise_plot_widget, stretch=1)

        self.ui.init_signal_vlayout.update()
        self.ui.noisy_signal_vlayout.update()
        self.ui.noise_distribution_vlayout.update()
        self.ui.noisePlotVLayout.update()

        self.update()

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

        self.ui.initialALabel.setText(f"a_init = {a}")
        self.ui.initialBLabel.setText(f"b_init = {b}")
        # self.ui.initialSigmaLabel.setText(f"sigma = {self.ui.sigmaSlider.value()}")

        self.initial_signal = [a * x + b for x in range(n)]

        self.plot_initial_signal()
        self.add_noise_to_signal()

    def plot_initial_signal(self):
        self.initial_signal_plot_widget.figure.clear()
        n_parameter = len(self.initial_signal)
        a_parameter = self.ui.aSlider.value()
        b_parameter = self.ui.bSlider.value()

        ax = self.initial_signal_plot_widget.figure.add_subplot(111)
        ax.scatter(range(n_parameter), self.initial_signal)
        ax.set_title("Initial Signal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        self.initial_signal_plot_widget.draw()

        self.ui.label_2.setText(f"A = {a_parameter}")
        self.ui.label_3.setText(f"B = {b_parameter}")
        self.ui.label_4.setText(f"N = {n_parameter}")

    def add_noise_to_signal(self):
        noise_type = self.ui.noiseType.currentText()
        sigma = int(self.ui.sigmaSlider.value())
        mean = int(self.ui.meanSlider.value())

        if noise_type == "Gaussian":
            self.noise = np.random.normal(sigma, mean, size=len(self.initial_signal))
            self.noisy_signal = self.initial_signal + self.noise

        else:
            low = mean - sigma * math.sqrt(3)
            high = mean + sigma * math.sqrt(3)
            self.noise = np.random.uniform(low, high, size=len(self.initial_signal))
            self.noisy_signal = self.initial_signal + self.noise

        self.find_linear_trend()
        self.plot_noisy_signal()
        self.ui.label_5.setText(f"Mean={mean}")
        self.ui.label_6.setText(f"S={sigma}")
        self.ui.initialSigmaLabel.setText(f"sigma_init = {sigma}")

    def plot_noisy_signal(self):
        self.noisy_signal_plot_widget.figure.clear()
        ax = self.noisy_signal_plot_widget.figure.add_subplot(111)
        ax.scatter(range(len(self.noisy_signal)), self.noisy_signal)
        ax.plot(
            range(len(self.noisy_signal)),
            self.estimated_a * np.arange(len(self.noisy_signal)) + self.estimated_b,
            color="orange",
        )
        ax.set_title("Noisy Signal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        self.noisy_signal_plot_widget.draw()

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

            self.ui.estimatedALabel.setText(f"a_est = {self.estimated_a:.3f}")
            self.ui.estimatedBLabel.setText(f"b_est = {self.estimated_b:.3f}")
            self.ui.estimatedSigmaLabel.setText(
                f"sigma_est = {self.estimated_sigma:.3f}"
            )
            self.calculate_error()
            self.plot_noise()
            self.plot_noise_distribution()

    def plot_noise(self):
        self.noise_plot_widget.figure.clear()

        ax = self.noise_plot_widget.figure.add_subplot(111)
        ax.plot(range(len(self.initial_signal)), self.noise)
        ax.set_title("Noise")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        self.noise_plot_widget.draw()

    def plot_noise_distribution(self):
        self.noise_distribution_plot_widget.figure.clear()
        distribution_type = self.ui.noiseType.currentText()
        ax = self.noise_distribution_plot_widget.figure.add_subplot(111)
        bins = int(self.ui.binsSlider.value())
        counts, bin_edges, patches = ax.hist(self.noise, bins, density=False)
        ax.hist(self.noise, bins)
        ax.set_title("Noise Distribution")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        total_count = np.sum(counts)
        bin_width = bin_edges[1] - bin_edges[0]

        if distribution_type == "Gaussian":
            mu, std = norm.fit(self.noise)
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            ax.plot(x, p * total_count * bin_width, "k", linewidth=2)
        elif distribution_type == "Uniform":
            xmin, xmax = ax.get_xlim()
            p = uniform.pdf(
                bin_edges,
                loc=np.min(self.noise),
                scale=np.max(self.noise) - np.min(self.noise),
            )
            ax.plot(bin_edges, p * total_count * bin_width, "k", linewidth=2)

        self.noise_distribution_plot_widget.draw()
        self.ui.label_7.setText(f"Bins = {bins}")

    def calculate_error(self):
        if self.estimated_a != 0 and self.estimated_b != 0:
            self.a_error = (
                np.abs(np.abs(self.estimated_a) - int(self.ui.aSlider.value()))
                / self.estimated_a
            )
            self.b_error = (
                np.abs(np.abs(self.estimated_b) - int(self.ui.bSlider.value()))
                / self.estimated_b
            )
            self.sigma_error = (
                np.abs(np.abs(self.estimated_sigma) - int(self.ui.sigmaSlider.value()))
                / self.estimated_sigma
            )
            a = self.ui.aSlider.value()
            b = self.ui.bSlider.value()
            sigma = self.ui.sigmaSlider.value()
            self.ui.errorALabel.setText(f"a_err = {self.a_error:.3f}")
            self.ui.errorBLabel.setText(f"b_err = {self.b_error:.3f}")
            self.ui.errorSigmaLabel.setText(f"sigma_err = {self.sigma_error:.3f}")

            predicted_signal = [
                self.estimated_a * x + self.estimated_b
                for x in range(self.ui.nSlider.value())
            ]

            slope, intercept, r_squared, p_value, std_err = linregress(
                self.noisy_signal, predicted_signal
            )
            print(r_squared)
            self.ui.RLabel.setText(f"R^2={r_squared:.3f}")


if __name__ == "__main__":
    import sys

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())
