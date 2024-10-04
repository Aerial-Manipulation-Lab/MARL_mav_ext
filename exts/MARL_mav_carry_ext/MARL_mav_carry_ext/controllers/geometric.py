import torch 
from omni.isaac.lab.utils.math import quat_inv, quat_mul

class GeometricController():
    """
    The geometric controller for the falcon drones

    """

    def __init__(self):
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

        self.gravity = torch.tensor([0.0, 0.0, -9.8066])

    # function to overwrite parameters from yaml file
    # function to check if all parameters are valid

    def getCommand(
        self, state: torch.tensor, 
        state_accelerations: torch.tensor, 
        actions: torch.tensor,
        reference_trajectory: torch.tensor):
        """
        Get the command for the drone
        inputs:
        state: current state of the drone given by Isaac sim: [pos, quat, lin_vel, ang_vel]
        state_accelerations: current accelerations of the drone given by Isaac sim: [lin_acc, ang_acc]
        actions: actions given by the policy, 4 rotor thrusts for each propeller
        reference_trajectory: reference trajectory for the drone: [pos, quat, lin_vel, ang_vel, lin_acc, ang_acc, jerk, snap]
        """

        # update low pass filters: not here for now

        body_acceleration = quat_inv(state[:, 3:7]) * (state_accelerations[:, 0:3] - self.gravity)

    

