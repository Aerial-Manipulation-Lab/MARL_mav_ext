import torch

from omni.isaac.lab.utils.math import (
    euler_xyz_from_quat,
    matrix_from_quat,
    normalize,
    quat_from_matrix,
    quat_inv,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
)


class GeometricController:
    """
    The geometric controller for the falcon drones

    """

    def __init__(self, num_envs: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_envs = num_envs

        self.p_err_max_ = torch.full((self.num_envs, 3), torch.finfo(torch.float32).max, device=self.device)
        self.v_err_max_ = torch.full(
            (
                self.num_envs,
                3,
            ),
            torch.finfo(torch.float32).max,
            device=self.device,
        )

        self.p_offset = torch.tensor([[0.0, 0.0, -0.0]] * self.num_envs, device=self.device)
        self.integration_max = torch.tensor([0.0, 0.0, 0.0], device=self.device)

        # sim and drone parameters
        self.gravity = torch.tensor([[0.0, 0.0, -9.8066]] * self.num_envs, device=self.device)
        self.z_i = torch.tensor([[0.0, 0.0, 1.0]] * self.num_envs, device=self.device)
        self.thrust_map = torch.tensor([1.562522e-06, 0.0, 0.0], device=self.device)
        self.torque_map = torch.tensor([1.908873e-08, 0.0, 0.0], device=self.device)
        self.t_last = 0.0
        self.falcon_mass = 0.617  # kg
        self.l = 0.075
        self._epsilon = torch.tensor(1e-6, device=self.device)  # avoid division by zero
        self.kappa = 0.022
        self.beta = torch.deg2rad(torch.tensor([45], device=self.device))
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
        self.G_1_inv = torch.linalg.pinv(self.G_1)
        self.inertia_mat = torch.diag(torch.tensor([0.00164, 0.00184, 0.0030], device=self.device))
        self.min_thrust = torch.tensor(0.0, device=self.device)
        self.max_thrust = torch.tensor(6.25, device=self.device)

        # controller parameters
        self.kp_acc = torch.tensor([4.0, 4.0, 9.0], device=self.device)
        self.kd_acc = torch.tensor([4.0, 4.0, 6.0], device=self.device)
        self.ki_acc = torch.tensor([0.0, 0.0, 0.0], device=self.device)

        self.kp_rate = torch.tensor([25.0, 25.0, 8.0], device=self.device)
        self.kp_att_xy = 150.0
        self.kp_att_z = 5.0

        # self.filter_sampling_frequency = torch.tensor([50, 50, 50], device=self.device)
        # self.filter_cutoff_frequency = torch.tensor([20, 20, 20], device=self.device)
        # self.filter_cutoff_frequency_bodyrate = 20

        # self.filter_acc = LowPassFilter(self.filter_cutoff_frequency, self.filter_sampling_frequency, self.gravity)

    # function to overwrite parameters from yaml file
    # function to check if all parameters are valid

    def getCommand(
        self,
        state: dict,
        actions: torch.tensor,
        setpoint: dict,
    ) -> torch.tensor:
        """
        Get the command for the drone
        inputs:
        state: current observed state of the drone given by Isaac sim: [pos, quat, lin_vel, ang_vel, lin_acc, ang_acc]
        actions: actions given by the policy, 4 rotor thrusts for each propeller
        setpoint: setpoint given by the policy [pos, lin_vel, lin_acc, quat, ang_vel, ang_acc]
        """

        # update low pass filters: not here for now

        # acceleration command TODO: implement aceleration low pass filter
        p_ref_cg = setpoint["pos"] - quat_rotate(state["quat"], self.p_offset)
        pos_error = torch.clamp(p_ref_cg - state["pos"], -self.p_err_max_, self.p_err_max_)
        vel_error = torch.clamp(setpoint["lin_vel"] - state["lin_vel"], -self.v_err_max_, self.v_err_max_)
        des_acc = self.kp_acc * pos_error + self.kd_acc * vel_error + setpoint["lin_acc"]
        # estimation of load acceleration in world frame
        current_collective_thrust = actions.sum(1)  # sum over all propellors
        acc_load = (
            state["lin_acc"] - self.gravity - quat_rotate(state["quat"], current_collective_thrust / self.falcon_mass)
        )
        acc_cmd = des_acc - self.gravity - acc_load
        z_b_des = normalize(acc_cmd)  # desired new thrust direction
        collective_thrust_des_magntiude = torch.norm(acc_cmd, dim=1, keepdim=True) * self.falcon_mass
        current_collective_thrust_magnitude = torch.norm(current_collective_thrust, dim=1, keepdim=True)

        # attitude command
        # Calculate the desired quaternion
        setpoint_yaw = setpoint["yaw"]
        # calculate intermediate axis and new desired body frame
        x_intermediate_des = torch.cat(
            (torch.cos(setpoint_yaw), torch.sin(setpoint_yaw), torch.zeros_like(setpoint_yaw)), dim=1
        )
        y_b_des = normalize(torch.linalg.cross(z_b_des, x_intermediate_des))  # / (
        x_b_des = torch.linalg.cross(y_b_des, z_b_des)

        # calculate the desired quaternion
        des_rot_matrix = torch.stack([x_b_des, y_b_des, z_b_des], dim=2)
        q_cmd = quat_from_matrix(des_rot_matrix)
        # angular velocity command
        # retrieve the current body axes of the drone
        current_rot_matrix = matrix_from_quat(state["quat"])
        x_b = current_rot_matrix[..., 0]
        y_b = current_rot_matrix[..., 1]
        z_b = current_rot_matrix[..., 2]

        T_dot = self.falcon_mass * torch.sum(setpoint["jerk"] * z_b, dim=-1, keepdim=True)
        h_omega = self.falcon_mass * setpoint["jerk"] - T_dot * z_b  # rotational derivative of z_b
        mask = (current_collective_thrust_magnitude > 0.01).squeeze()  # avoid division by zero
        h_omega[mask] /= current_collective_thrust_magnitude[mask]
        omega_b_x = (-h_omega * y_b).sum(-1, keepdim=True)
        omega_b_y = (h_omega * x_b).sum(-1, keepdim=True)
        omega_b_z = setpoint["yaw_rate"] * (self.z_i * z_b).sum(-1, keepdim=True)
        omega_b_ref = torch.cat((omega_b_x, omega_b_y, omega_b_z), dim=-1)

        # tilt prioritized attitude control
        quat_diff = quat_mul(quat_inv(state["quat"]), q_cmd)
        q_e_w = quat_diff[..., 0].view(self.num_envs, 1)
        q_e_x = quat_diff[..., 1].view(self.num_envs, 1)
        q_e_y = quat_diff[..., 2].view(self.num_envs, 1)
        q_e_z = quat_diff[..., 3].view(self.num_envs, 1)
        norm_factor = (2 / (torch.sqrt(q_e_w.square() + q_e_z.square())) + self._epsilon).view(self.num_envs, 1)
        zeros = torch.zeros((self.num_envs, 1), device=self.device)
        q_e_red = norm_factor * torch.cat((q_e_w * q_e_x - q_e_y * q_e_z, q_e_w * q_e_y + q_e_x * q_e_z, zeros), dim=-1)
        q_e_yaw = norm_factor * torch.cat([zeros, zeros, q_e_z], dim=-1)
        ang_vel_body = quat_rotate(quat_inv(state["quat"]), state["ang_vel"])
        alpha_b_des = (
            self.kp_att_xy * q_e_red
            + self.kp_att_z * torch.sign(q_e_w) * q_e_yaw
            + self.kp_rate * (omega_b_ref - ang_vel_body)
        )

        # calculate the thrust per propellor
        # inertia * alpha_cmd + omega x inertia * omega = torque = G * thrusts
        product = self.inertia_mat.matmul(alpha_b_des.transpose(0, 1)).transpose(0, 1) + torch.linalg.cross(
            ang_vel_body, self.inertia_mat.matmul(ang_vel_body.transpose(0, 1)).transpose(0, 1)
        )
        rh_side = torch.cat((collective_thrust_des_magntiude, product), dim=-1)
        thrusts = self.G_1_inv.matmul(rh_side.transpose(0, 1)).transpose(0, 1)
        thrusts = torch.clamp(thrusts, self.min_thrust, self.max_thrust)

        return thrusts, acc_cmd, q_cmd
