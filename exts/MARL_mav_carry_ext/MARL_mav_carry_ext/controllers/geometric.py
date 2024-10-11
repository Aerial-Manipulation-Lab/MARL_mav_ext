import torch 
from omni.isaac.lab.utils.math import quat_inv, quat_mul, quat_rotate, quat_from_matrix, matrix_from_quat
from torch.nn.functional import normalize
from .utils import LowPassFilter

class GeometricController():
    """
    The geometric controller for the falcon drones

    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.drag_compensation = False # no aerodynamics in the sim yet
        self.load_compensation = True # compensate for acceleration due to payload
        self.use_bodyrate_ref = False

        self.p_err_max_ = torch.full((3,), torch.finfo(torch.float32).max).to(self.device)
        self.v_err_max_ = torch.full((3,), torch.finfo(torch.float32).max).to(self.device)

        self.p_offset = torch.tensor([0.0, 0.0, 0.0]).to(self.device)
        self.integration_max = torch.tensor([0.0, 0.0, 0.0]).to(self.device)

        # sim and drone parameters
        self.gravity = torch.tensor([0.0, 0.0, -9.8066]).to(self.device)
        self.thrust_map = torch.tensor([1.562522e-06, 0.0, 0.0]).to(self.device)
        self.torque_map = torch.tensor([1.908873e-08, 0.0, 0.0]).to(self.device)
        self.t_last = 0.0
        self.falcon_mass = 1.2336 # kg
        self.l = 0.075
        self.kappa = self.torque_map[0]/self.thrust_map[0]
        self.beta = torch.deg2rad(torch.tensor([45], device=self.device))
        self.G_1 = torch.tensor([[1, 1, 1, 1],
                                 [self.l * torch.sin(self.beta), -self.l * torch.sin(self.beta), -self.l * torch.sin(self.beta), self.l * torch.sin(self.beta)],
                                 [-self.l * torch.cos(self.beta), -self.l * torch.cos(self.beta), self.l * torch.cos(self.beta), self.l * torch.cos(self.beta)],
                                 [self.kappa, -self.kappa, self.kappa, -self.kappa]], device=self.device)
        self.inertia_mat = torch.diag(torch.tensor([0.00164, 0.00184, 0.0030], device=self.device))
        self.min_thrust = torch.tensor(0.0, device=self.device)
        self.max_thrust = torch.tensor(25.0/4, device=self.device)

        # controller parameters
        self.kp_acc = torch.tensor([4.0, 4.0, 9.0]).to(self.device)
        self.kd_acc = torch.tensor([4.0, 4.0, 6.0]).to(self.device)
        self.ki_acc = torch.tensor([0.0, 0.0, 0.0]).to(self.device)

        self.kp_rate = torch.tensor([35.0, 35.0, 8.0]).to(self.device)
        self.kp_att_xy = 150.0
        self.kp_att_z = 5.0
        
        self.filter_sampling_frequency = torch.tensor([50, 50, 50], device=self.device)
        self.filter_cutoff_frequency = torch.tensor([20, 20, 20], device=self.device)
        self.filter_cutoff_frequency_bodyrate = 20

        # self.filter_acc = LowPassFilter(self.filter_cutoff_frequency, self.filter_sampling_frequency, self.gravity)

    # function to overwrite parameters from yaml file
    # function to check if all parameters are valid

    def getCommand(
        self, 
        state: dict, 
        actions: torch.tensor,
        setpoint: dict,) -> torch.tensor:
        """
        Get the command for the drone
        inputs:
        state: current observed state of the drone given by Isaac sim: [pos, quat, lin_vel, ang_vel, lin_acc, ang_acc]
        actions: actions given by the policy, 4 rotor thrusts for each propeller
        setpoint: setpoint given by the spline [pos, lin_vel, lin_acc, quat, ang_vel, ang_acc]
        """

        # update low pass filters: not here for now

        # acceleration command TODO: implement aceleration low pass filter
        p_ref_cg = setpoint["pos"] - quat_rotate(state["quat"].unsqueeze(0), self.p_offset.unsqueeze(0))[0]

        pos_error = torch.clamp(p_ref_cg - state["pos"], -self.p_err_max_, self.p_err_max_)
        vel_error = torch.clamp(setpoint["lin_vel"] - state["lin_vel"], -self.v_err_max_, self.v_err_max_)

        des_acc = self.kp_acc * pos_error + self.kd_acc * vel_error + setpoint["lin_acc"]
        # estimation of load acceleration in world frame
        current_collective_thrust = actions.sum(0)
        acc_load = state["lin_acc"] - self.gravity - quat_rotate(state["quat"].unsqueeze(0), current_collective_thrust.unsqueeze(0)/self.falcon_mass)
        # acc_load_filtered = self.filter_acc.add(acc_load).unsqueeze(0)
        des_thrust = self.falcon_mass * (des_acc - self.gravity + acc_load)
        z_b_des = normalize(des_thrust)[0] # desired new thrust direction
        collective_thrust_des_magntiude = torch.norm(des_thrust)
        current_collective_thrust_magnitude = torch.norm(current_collective_thrust)

        # attitude command
        # Calculate the desired quaternion
        setpoint_yaw = setpoint["yaw"]
        # calculate intermediate axis and new desired body frame
        x_intermediate_des = torch.tensor([torch.cos(setpoint_yaw), torch.sin(setpoint_yaw), 0.0], device=self.device)
        y_b_des = torch.linalg.cross(z_b_des, x_intermediate_des)/torch.norm(torch.linalg.cross(z_b_des, x_intermediate_des))
        x_b_des = torch.linalg.cross(y_b_des, z_b_des)

        # calculate the desired quaternion
        des_rot_matrix = torch.stack([x_b_des, y_b_des, z_b_des])
        q_cmd = quat_from_matrix(des_rot_matrix.unsqueeze(0))
 
        # angular velocity command
        # retrieve the current body axes of the drone
        curent_rot_matrix = matrix_from_quat(state["quat"].unsqueeze(0))[0]
        x_b = curent_rot_matrix[:, 0]
        y_b = curent_rot_matrix[:, 1]
        z_b = curent_rot_matrix[:, 2]

        z_i = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        T_dot = self.falcon_mass * setpoint["jerk"].dot(z_b)
        h_omega = (self.falcon_mass * setpoint["jerk"] - T_dot * z_b) # rotational derivative of z_b
        if current_collective_thrust_magnitude > 0.01: # avoid division by zero
            h_omega /= current_collective_thrust_magnitude
        omega_b_ref = torch.tensor([-h_omega.dot(y_b), h_omega.dot(x_b), setpoint["yaw"] * (z_i.dot(z_b))], device=self.device) # TODO: needs to be yaw_dot, for now yaw = 0 = yaw_dot
        
        # angular acceleration command
        T_ddot = self.falcon_mass * setpoint["snap"].dot(z_b) + self.falcon_mass * h_omega.dot(setpoint["jerk"])
        if current_collective_thrust_magnitude > 0.01: # avoid division by zero
            h_alpha = (self.falcon_mass/current_collective_thrust_magnitude) * setpoint["snap"] - (torch.linalg.cross(state["ang_vel"],h_omega) \
                + (2*T_dot/current_collective_thrust_magnitude) * h_omega + T_ddot/current_collective_thrust_magnitude * z_b)
        else:
            h_alpha = (self.falcon_mass) * setpoint["snap"] - (torch.linalg.cross(state["ang_vel"],h_omega) \
                + (2*T_dot) * h_omega + T_ddot * z_b)

        alpha_b_ref = torch.tensor([-h_alpha.dot(y_b), h_alpha.dot(x_b), setpoint["yaw"] * (z_i.dot(z_b))], device=self.device) # TODO: needs to be yaw_ddot, for now yaw = 0 = yaw_ddot

        # tilt prioritized attitude control
        quat_diff = quat_mul(q_cmd, quat_inv(state["quat"].unsqueeze(0)))[0]
        q_e_w = quat_diff[0]
        q_e_x = quat_diff[1]
        q_e_y = quat_diff[2]
        q_e_z = quat_diff[3]
        norm_factor = 1/(torch.sqrt(q_e_w.square() + q_e_z.square()))
        q_e_red = norm_factor * torch.tensor([q_e_w*q_e_x - q_e_y*q_e_z, q_e_w*q_e_y + q_e_x*q_e_z, 0.0], device=self.device)
        q_e_yaw = norm_factor * torch.tensor([0.0, 0.0, q_e_z], device=self.device)

        alpha_b_des = self.kp_att_xy * q_e_red + self.kp_att_z * q_e_yaw + self.kp_rate * (omega_b_ref - state["ang_vel"]) + alpha_b_ref

        # calculate the thrust per propellor
        # intertia * alpha_cmd + omega x inertia * omega = torque = G * thrusts
        product = self.inertia_mat.mv(alpha_b_des) + torch.linalg.cross(state["ang_vel"],(self.inertia_mat.mv(state["ang_vel"])))
        coll_thrust_tensor = torch.tensor([[collective_thrust_des_magntiude]], device=self.device)
        rh_side = torch.cat((coll_thrust_tensor[0], product), dim=0)
        thrusts = torch.inverse(self.G_1).mv(rh_side)

        thrusts = torch.max(self.min_thrust, torch.min(thrusts, self.max_thrust))

        return thrusts


