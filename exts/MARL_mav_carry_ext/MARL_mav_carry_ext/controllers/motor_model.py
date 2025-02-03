import torch

class RotorMotor:
    def __init__(self, num_envs: int, init_omega: torch.Tensor):
        # motor parameters
        self.num_envs = num_envs
        self.motor_omega_min = 150.0
        self.motor_omega_max = 2800.0
        self.tau_up = 0.033
        self.tau_down = 0.033
        self.motor_inertia = 9.3575e-6
        
        # rotor parameters
        self.thrust_map = torch.tensor([1.562522e-06, 0.0, 0.0])
        self.torque_map = torch.tensor([3.4375484e-08, 0.0, 0.0])

        self.current_omega = init_omega
        self.direction = torch.tensor([1.,  -1., 1., -1., 1.,  -1., 1., -1.,  1.,  -1., 1., -1.], device="cuda")

    def get_motor_thrusts_moments(self, target_rates: torch.Tensor, sampling_time: float):
        """
        Compute filtered thrusts and angular velocities for the motor.

        Args:
            target_rates: Desired rotational rate (omega) from the controller.
            sampling_time: Time interval between two consecutive calls.

        Returns:
            filtered_thrusts: Filtered thrust values.
        """
        # Check if body rate is going up or down
        tau = torch.where(target_rates > self.current_omega, self.tau_up, self.tau_down)
        alpha = torch.exp(-sampling_time / tau)
        print("current_omega: ", self.current_omega)

        # Update current omega
        self.current_omega = alpha * self.current_omega + (1 - alpha) * target_rates

        # Compute thrusts based on current omega
        thrusts = self.thrust_map[0] * self.current_omega**2

        # Compute moments based on current omega
        moments = self.torque_map[0] * self.current_omega**2 * self.direction

        return thrusts, moments