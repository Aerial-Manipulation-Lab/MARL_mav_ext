import torch

from MARL_mav_carry_ext.controllers.utils import LowPassFilter

from isaaclab.utils.math import quat_inv, quat_rotate


class IndiController:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # allocation matrix TODO: define this in a common place with the geometric controller
        self.kappa = 0.022
        self.beta = torch.deg2rad(torch.tensor([45], device=self.device))
        self.l = 0.10606601717798213
        self.G_1 = torch.tensor(
            [
                [1, 1, 1, 1],
                [
                    self.l * torch.sin(self.beta),
                    -self.l * torch.sin(self.beta),
                    -self.l * torch.sin(self.beta),
                    self.l * torch.sin(self.beta),
                ],
                [
                    -self.l * torch.cos(self.beta),
                    -self.l * torch.cos(self.beta),
                    self.l * torch.cos(self.beta),
                    self.l * torch.cos(self.beta),
                ],
                [self.kappa, -self.kappa, self.kappa, -self.kappa],
            ],
            device=self.device,
        )
        self.G_1_inv = torch.linalg.inv(self.G_1)

        self.motor_inertia_z = 9.3575e-6  # [kgm^2]
        self.G_2 = torch.zeros((4, 4), device=self.device)
        self.G_2[3, :] = torch.tensor(
            [self.motor_inertia_z, -self.motor_inertia_z, self.motor_inertia_z, -self.motor_inertia_z],
            device=self.device,
        )
        self.motor_omega_min = 150.0  # [rad/s]
        self.motor_omega_max = 2800.0  # [rad/s]
        self.thrust_min = 0.0
        self.thrust_max = 6.25  # [N]
        self.thrust_min_collective = 0.0
        self.thrust_max_collective = self.thrust_max * 4  # [N]
        self.thrust_map = torch.tensor([1.562522e-06, 0.0, 0.0], device=self.device)

        self.falcon_mass = 0.6017  # kg
        self.inertia_mat = torch.diag(torch.tensor([0.00164, 0.00184, 0.0030], device=self.device))
        self.rope_offset = -0.03
        self.p_offset = torch.tensor([[0.0, 0.0, self.rope_offset]] * self.num_envs, device=self.device)

        # TODO REMOVE
        # self.kp = torch.tensor([100.0, 100.0, 10.0], device=self.device)

        # low pass filters
        self.filter_sampling_frequency = torch.full(
            (self.num_envs, 1), 300.0, device=self.device
        )  # filter frequency, same as control frequency (Hz)
        self.filter_cutoff_frequency = torch.full(
            (self.num_envs, 1), 12.0, device=self.device
        )  # accelerometer filter cut-off frequency (Hz)
        self.filter_init_value_mot = torch.full((self.num_envs, 4), 0.0, device=self.device)
        self.filter_init_value_rate = torch.full((self.num_envs, 3), 0.0, device=self.device)

        self.filterMot_ = LowPassFilter(
            self.filter_cutoff_frequency, self.filter_sampling_frequency, self.filter_init_value_mot
        )
        self.filterRate_ = LowPassFilter(
            self.filter_cutoff_frequency, self.filter_sampling_frequency, self.filter_init_value_rate
        )

        self.debug = True
        if self.debug:
            self.filtered_ang_acc = torch.zeros((self.num_envs, 3), device=self.device)
            self.unfiltered_mot = torch.zeros((self.num_envs, 4), device=self.device)
            self.filtered_mot = torch.zeros((self.num_envs, 4), device=self.device)

    def getCommand(
        self,
        state: dict,
        actions: torch.tensor,
        alpha_cmd: torch.tensor,
        acc_cmd: torch.tensor,
        acc_load: torch.tensor,
    ) -> torch.tensor:
        forces = actions.sum(-1)
        filtered_forces = self.filterMot_.add(forces)
        self.filterRate_.add(state["ang_vel"])
        ang_acc_filtered = self.filterRate_.derivative()

        if self.debug:
            self.filtered_ang_acc = ang_acc_filtered
            self.unfiltered_mot = forces
            self.filtered_mot = filtered_forces

        omega = quat_rotate(quat_inv(state["quat"]), state["ang_vel"])  # body rates # normally from IMU
        omega_dot = quat_rotate(
            quat_inv(state["quat"]), state["ang_acc"]
        )  # body accelerations # normally from derivative filtered body rate
        tau = torch.matmul(self.G_1, forces.transpose(0, 1)).transpose(0, 1)[:, 1:]  # torque commands
        mu = torch.zeros((self.num_envs, 4), device=self.device)
        collective_thrust_des_magntiude = torch.norm(acc_cmd, dim=1) * self.falcon_mass
        mu[:, 0] = torch.clamp(collective_thrust_des_magntiude, self.thrust_min_collective, self.thrust_max_collective)
        mu_ndi = mu

        moments = self.inertia_mat.matmul(alpha_cmd.transpose(0, 1)).transpose(0, 1) + torch.linalg.cross(
            omega, self.inertia_mat.matmul(omega.transpose(0, 1)).transpose(0, 1)
        )  # - torch.linalg.cross(
        # self.p_offset, quat_rotate(quat_inv(state["quat"]), acc_load * self.falcon_mass)) # M_load in body frame

        mu_ndi[:, 1:] = moments
        mu[:, 1:] = tau + self.inertia_mat.matmul((alpha_cmd - omega_dot).transpose(0, 1)).transpose(0, 1)

        # without heading control
        mu[:, 3] = mu_ndi[:, 3]
        thrusts = self.G_1_inv.matmul(mu.transpose(0, 1))
        thrusts = torch.clamp(thrusts, self.thrust_min, self.thrust_max)

        rotor_speeds = torch.sqrt(thrusts / self.thrust_map[0]).transpose(0, 1)

        return rotor_speeds

    # def getCommand(
    #     self,
    #     state: dict,
    #     actions: torch.tensor,
    #     setpoint: dict,
    # ) -> torch.tensor:
    #     forces = actions.sum(-1)
    #     # filtered_forces = self.filterMot_.add(forces)
    #     # self.filterRate_.add(state["ang_vel"])
    #     # ang_acc_filtered = self.filterRate_.derivative()

    #     # if self.debug:
    #     #     self.filtered_ang_acc = ang_acc_filtered
    #     #     self.unfiltered_mot = forces
    #     #     self.filtered_mot = filtered_forces

    #     omega = quat_rotate(quat_inv(state["quat"]), state["ang_vel"])  # body rates # normally from IMU
    #     alpha_cmd = self.kp * (setpoint["body_rates"] - omega)
    #     acc_cmd = setpoint["cthrust"]
    #     omega_dot = quat_rotate(
    #         quat_inv(state["quat"]), state["ang_acc"]
    #     )  # body accelerations # normally from derivative filtered body rate
    #     tau = torch.matmul(self.G_1, forces.transpose(0, 1)).transpose(0, 1)[:, 1:]  # torque commands
    #     mu = torch.zeros((self.num_envs, 4), device=self.device)
    #     # collective_thrust_des_magntiude = torch.norm(acc_cmd, dim=1) * self.falcon_mass
    #     collective_thrust_des_magntiude = acc_cmd * self.falcon_mass
    #     mu[:, 0] = torch.clamp(collective_thrust_des_magntiude, self.thrust_min_collective, self.thrust_max_collective)
    #     mu_ndi = mu

    #     moments = self.inertia_mat.matmul(alpha_cmd.transpose(0, 1)).transpose(0, 1) + torch.linalg.cross(
    #         omega, self.inertia_mat.matmul(omega.transpose(0, 1)).transpose(0, 1)
    #     )  # - torch.linalg.cross(
    #     # self.p_offset, quat_rotate(quat_inv(state["quat"]), acc_load * self.falcon_mass)) # M_load in body frame

    #     mu_ndi[:, 1:] = moments
    #     mu[:, 1:] = tau + self.inertia_mat.matmul((alpha_cmd - omega_dot).transpose(0, 1)).transpose(0, 1)

    #     # without heading control
    #     mu[:, 3] = mu_ndi[:, 3]
    #     thrusts = self.G_1_inv.matmul(mu.transpose(0, 1))
    #     thrusts = torch.clamp(thrusts, self.thrust_min, self.thrust_max)

    #     rotor_speeds = torch.sqrt(thrusts / self.thrust_map[0]).transpose(0, 1)

    #     return rotor_speeds

    def reset(self, env_ids):
        self.filterMot_.reset(env_ids)
        self.filterRate_.reset(env_ids)
        # pass
