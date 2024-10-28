import math
import torch


class LowPassFilter:
    def __init__(self, cutoff_frequency, sampling_frequency, initial_value):
        """
        cutoff_frequency: torch.Tensor of shape (rows,)
        sampling_frequency: torch.Tensor of shape (rows,)
        initial_value: torch.Tensor of shape (rows,)
        """
        self.sampling_frequency = sampling_frequency
        self.denominator = self.init_den(cutoff_frequency, sampling_frequency)
        self.numerator = self.init_num(cutoff_frequency, sampling_frequency)

        # Store input and output history (2-step)
        self.input = initial_value.unsqueeze(1).repeat(1, 2)  # (rows, 2)
        self.output = initial_value.unsqueeze(1).repeat(1, 2)  # (rows, 2)

    def init_den(self, fc, fs):
        K = torch.tan(math.pi * fc / fs)
        poly = K**2 + math.sqrt(2) * K + 1.0

        denominator = torch.zeros_like(fc).unsqueeze(1).repeat(1, 2)  # (rows, 2)
        denominator[:, 0] = 2.0 * (K**2 - 1.0) / poly
        denominator[:, 1] = (K**2 - math.sqrt(2) * K + 1.0) / poly

        return denominator

    def init_num(self, fc, fs):
        K = torch.tan(math.pi * fc / fs)
        poly = K**2 + math.sqrt(2) * K + 1.0

        numerator = torch.zeros_like(fc).unsqueeze(1).repeat(1, 2)  # (rows, 2)
        numerator[:, 0] = K**2 / poly
        numerator[:, 1] = 2.0 * numerator[:, 0]

        return numerator

    def add(self, sample):
        """
        sample: torch.Tensor of shape (rows,)
        Returns filtered output.
        """
        # Shift the input and output history
        self.input[:, 1] = self.input[:, 0]
        self.input[:, 0] = sample

        # Compute the filter output
        out = self.numerator[:, 0] * self.input[:, 1] + (
            self.numerator * self.input - self.denominator * self.output
        ).sum(dim=1)

        # Shift the output history
        self.output[:, 1] = self.output[:, 0]
        self.output[:, 0] = out
        print("new value added")
        return out

    def derivative(self):
        """
        Returns the derivative of the filtered signal.
        """
        return self.sampling_frequency * (self.output[:, 0] - self.output[:, 1])


# # Example usage
# cutoff_frequency = torch.tensor([1.0, 0.5])  # Example: 2D input with different cutoffs
# sampling_frequency = torch.tensor([100.0, 100.0])  # Sampling at 100Hz for both
# initial_value = torch.tensor([0.0, 0.0])  # Starting with 0 for both dimensions

# lpf = LowPassFilter(cutoff_frequency=cutoff_frequency,
#                     sampling_frequency=sampling_frequency,
#                     initial_value=initial_value)

# # Simulate some input signal (multi-dimensional)
# input_signal = torch.tensor([[1.0, 0.9], [0.8, 0.7], [0.6, 0.5], [0.4, 0.3], [0.2, 0.1]])

# # Apply the low-pass filter to the signal
# for signal in input_signal:
#     filtered_signal = lpf.add(signal)
#     print("Filtered Output:", filtered_signal)
#     print("Derivative:", lpf.derivative())
