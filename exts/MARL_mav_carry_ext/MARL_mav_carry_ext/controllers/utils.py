import math
import torch


class LowPassFilter:
    def __init__(self, fc, fs, initial_value):
        self.sampling_freq = fs  # envs x 1
        self.num = self.init_num(fc, fs)  # envs x 1 x 2
        self.dem = self.init_dem(fc, fs)
        self.initial_value = initial_value
        self.input = initial_value.unsqueeze(2).repeat(1, 1, 2)  # envs x dim x 2
        self.output = initial_value.unsqueeze(2).repeat(1, 1, 2)  # envs x dim x 2

    def init_dem(self, fc, fs):
        K = torch.tan(math.pi * fc / fs)
        poly = K * K + math.sqrt(2.0) * K + 1.0
        dem = torch.zeros_like(fc).repeat(1, 2)
        dem[:, 0] = (2.0 * (K * K - 1.0) / poly).squeeze(1)
        dem[:, 1] = ((K * K - math.sqrt(2.0) * K + 1.0) / poly).squeeze(1)

        return dem.unsqueeze(1)

    def init_num(self, fc, fs):
        K = torch.tan(math.pi * fc / fs)
        poly = K * K + math.sqrt(2.0) * K + 1.0
        num = torch.zeros_like(fc).repeat(1, 2)
        num[:, 0] = (K * K / poly).squeeze(1)
        num[:, 1] = 2.0 * num[:, 0]

        return num.unsqueeze(1)

    def derivative(self):
        return self.sampling_freq * (self.output[:, :, 0] - self.output[:, :, 1])

    def add(self, sample):
        x2 = self.input[:, :, 1]
        self.input[:, :, 1] = self.input[:, :, 0]

        self.input[:, :, 0] = sample  # envs x dim x 1
        out = self.num[:, :, 0] * x2 + (self.num * self.input - self.dem * self.output).sum(dim=2)
        self.output[:, :, 1] = self.output[:, :, 0]
        self.output[:, :, 0] = out

        return out

    def valid(self):
        return (
            torch.isfinite(self.sampling_freq).all()
            and torch.isfinite(self.dem).all()
            and torch.isfinite(self.num).all()
            and torch.isfinite(self.input).all()
            and torch.isfinite(self.output).all()
        )

    def reset(self, env_ids):
        self.input[env_ids] = self.initial_value.unsqueeze(2).repeat(1, 1, 2)[env_ids]  # envs x dim x 2
        self.output[env_ids] = self.initial_value.unsqueeze(2).repeat(1, 1, 2)[env_ids]  # envs x dim x 2

    def __call__(self):
        return self.output[:, :, 0]
