import torch 
from omni.isaac.lab.utils.math import quat_inv, quat_mul, euler_xyz_from_quat
from torch.nn.functional import normalize

def get_yaw(self, yaw):
    x_b = quat_mul(self.q, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    x_proj = torch.tensor([x_b[0], x_b[1], 0.0])
    if torch.norm(x_proj) < 1e-3:
        return yaw
    x_proj_norm = normalize(x_proj)
    cross = torch.cross(torch.tensor([1.0, 0.0, 0.0]), x_proj_norm)
    angle = torch.asin(cross[2])
    if x_proj_norm[0] >= 0.0:
        return angle
    if x_proj_norm[1] >= 0.0:
        return torch.tensor(math.pi) - angle
    return -torch.tensor(math.pi) - angle

    def tiltPrioritizedControl(q, q_des):
        # Attitude control method from Fohn 2020.
        q_e = quat_inv(q) * q_des

        T_att = torch.tensor([[self.kp_att_xy, 0.0, 0.0],
                              [0.0, self.kp_att_xy, 0.0],
                              [0.0, 0.0, self.kp_att_z]])

        tmp = torch.tensor([q_e[0] * q_e[1] - q_e[2] * q_e[3],
                            q_e[0] * q_e[2] + q_e[1] * q_e[3],
                            q_e[3]])

        if q_e[0] <= 0:
            tmp[2] *= -1.0

        rate_cmd = 2.0 / torch.sqrt(q_e[0] * q_e[0] + q_e[3] * q_e[3]) * torch.matmul(T_att, tmp)

        return rate_cmd

class GeometricController():
    """
    The geometric controller for the falcon drones

    """

    def __init__(self):

        # controller parameters
        self.kp_acc = [4.0, 4.0, 9.0]
        self.kd_acc = [4.0, 4.0, 6.0]
        self.ki_acc = [0.0, 0.0, 0.0]

        self.kp_rate = [25.0, 25.0, 8.0]
        self.kp_att_xy = 150
        self.kp_att_z = 5
        
        self.filter_sampling_frequency = 300
        self.filter_cutoff_frequency = 6
        self.filter_cutoff_frequency_bodyrate = 20

        # self.drag_compensation = False # no aerodynamics in the sim yet
        self.load_compensation = True # compensate for acceleration due to payload
        self.use_bodyrate_ref = False

        self.p_err_max_ = torch.full((3,), torch.finfo(torch.float64).max)
        self.v_err_max_ = torch.full((3,), torch.finfo(torch.float64).max)

        self.p_offset = [0.0, 0.0, 0.0]
        self.integration_max = [0.0, 0.0, 0.0]

        # sim and drone parameters
        self.gravity = torch.tensor([0.0, 0.0, -9.8066])
        self.thrust_map = torch.tensor([1.562522e-06, 0.0, 0.0])
        self.t_last = 0.0
        self.falcon_mass = 1.2336 # kg

    # function to overwrite parameters from yaml file
    # function to check if all parameters are valid

    def getCommand(
        self, 
        drone_states: dict, 
        actions: torch.tensor,
        reference_trajectory: torch.tensor):
        """
        Get the command for the drone
        inputs:
        state: current state of the drone given by Isaac sim: [pos, quat, lin_vel, ang_vel]
        state_accelerations: current accelerations of the drone given by Isaac sim: [lin_acc, ang_acc]
        actions: actions given by the policy, 4 rotor thrusts for each propeller * 3 drones
        reference_trajectory: reference trajectory for the drone: [pos, quat, lin_vel, ang_vel, lin_acc, ang_acc, jerk, snap]
        """

        setpoints = reference_trajectory[0] # set point for each of the 3 drones

        # update low pass filters: not here for now

        body_acceleration = quat_inv(drone_states["quat"]) * (drone_states["lin_acc"] - self.gravity)
        thrust_f = self.thrust_map[0] * torch.matmul(actions.transpose(), actions)

        # acceleration command
        p_ref_cg = setpoint[:, 0:3] - drone_states["quat"] * self.p_offset

        pos_error = torch.clamp(p_ref_cg - drone_states["pos"], -self.p_err_max_, self.p_err_max_)
        vel_error = torch.clamp(setpoint[:, 3:6] - drone_states["lin_vel"], -self.v_err_max_, self.v_err_max_)

        if self.t_last > 0.0:
            dt = drone_states["time"] - t_last
            pos_err_int += pos_error * dt
            pos_err_int = torch.clamp(pos_err_int, -integration_max, integration_max)

        self.t_last = drone_states["time"]

        acc_cmd = self.kp_acc * pos_error + self.kd_acc * vel_error + self.ki_acc * pos_err_int + setpoint[:, 6:9] - self.gravity

        acc_cmd_integral = self.ki_acc * pos_err_int

        acc_load = drone_states["lin_acc"] - self.gravity - (actions * self.thrust_map[0] * torch.tensor([0.0, 0.0, 1.0])) / self.falcon_mass
        acc_cmd += acc_load
        thrust_command = torch.norm(acc_cmd) * self.falcon_mass

        # attitude command
        # Calculate the desired quaternion
        roll, pitch, setpoint_yaw = euler_xyz_from_quat(setpoint[:, 3:7])
        q_c = torch.tensor([0.0, 0.0, torch.sin(get_yaw(setpoint_yaw) / 2), torch.cos(get_yaw(setpoint_yaw) / 2)])
        y_c = quat_mul(q_c, torch.tensor([0.0, 1.0, 0.0, 0.0]))
        z_B = normalize(acc_cmd)
        x_B = normalize(torch.cross(y_c, z_B))
        y_B = normalize(torch.cross(z_B, x_B))
        R_W_B = torch.stack([x_B, y_B, z_B])
        q_des = torch.tensor([0.0, x_B[0], y_B[0], z_B[0]])

        q_cmd = q_des

        # angular acceleration command
        omega_cmd = tiltPrioritizedControl(drone_states.q(), q_cmd)

        if self.use_bodyrate_ref:
            alpha_cmd = omega_cmd + self.kp_rate * (drone_states["quat"].inverse() * setpoint[:, 9:12] - drone_states["quat"] * drone_states["ang_vel"])
        else:
            bx = quat_mul(drone_states["quat"], torch.tensor([1.0, 0.0, 0.0]))
            by = quat_mul(drone_states["quat"], torch.tensor([0.0, 1.0, 0.0]))
            bz = quat_mul(drone_states["quat"], torch.tensor([0.0, 0.0, 1.0]))
            hw = self.falcon_mass * (setpoint[:, 12:15] - torch.dot(bz, setpoint[:, 12:15]) * bz) # jerk
            if thrust_f >= 0.01:
                hw /= thrust_f
            w_ref = torch.tensor([-torch.dot(hw, by), torch.dot(hw, bx), setpoint[:, 9]])

            alpha_cmd = omega_cmd + self.kp_rate * (w_ref - drone_states["quat"] * drone_states["ang_vel"]) # for now unfiltered angular vel

        # calculate the thrust per propellor
        # intertia * alpha_cmd + omega x inertia * omega = torque = G * thrusts
        
        return thrusts


