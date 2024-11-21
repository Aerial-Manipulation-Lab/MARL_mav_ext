"""Helper class to plot results of ManagerBasedRLEnv"""
import matplotlib.pyplot as plt
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg
import math 
import torch

class ManagerBasedPlotter():
    def __init__(self, env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):

        # environment
        self.command_name = command_name
        self.env = env
        self.robot = env.scene[asset_cfg.name]
        self.load_id = self.robot.find_bodies("load_link")[0]
        self.drone_idx = self.robot.find_bodies("Falcon.*base_link")[0]
        self.sim_dt = env.sim.get_rendering_dt()

        # buffers
        self.metrics : dict = {}
        self.load_data :dict = {}
        self.drone_data_by_id : dict = {}

    def collect_metrics(self):
        """Collect the metrics from the command manager in the environment."""
        if not self.metrics:
            self.metrics = self.env.command_manager._terms[self.command_name].metrics.copy()
            for key in self.metrics:
                self.metrics[key] = self.metrics[key].tolist()
        else:
            for key in self.metrics:
                self.metrics[key].append(self.env.command_manager._terms[self.command_name].metrics[key].item())

    def collect_load_data(self):
        """Collect the load data from the environment."""
        # load data
        load_pos = self.robot.data.body_state_w[:, self.load_id, :3].squeeze(1)[0]
        load_orientation = self.robot.data.body_state_w[:, self.load_id, 3:7].squeeze(1)[0]
        load_vel = self.robot.data.body_state_w[:, self.load_id, 7:10].squeeze(1)[0]
        load_ang_vel = self.robot.data.body_state_w[:, self.load_id, 10:].squeeze(1)[0]
        load_acc = self.robot.data.body_state_w[:, self.load_id, 10:].squeeze(1)[0]
        load_ang_acc = self.robot.data.body_state_w[:, self.load_id, 10:].squeeze(1)[0]

        # references
        load_pos_ref = self.env.command_manager._terms[self.command_name].pose_command_w[..., :3][0]
        load_orientation_ref = self.env.command_manager._terms[self.command_name].pose_command_w[..., 3:7][0]
        load_vel_ref = self.env.command_manager._terms[self.command_name].twist_command_b[..., 0:3][0]
        load_ang_vel_ref = self.env.command_manager._terms[self.command_name].twist_command_b[..., 3:][0]

        # get the first point from the commanded trajectory
        if load_pos_ref.shape[0] > 1:
            load_pos_ref = load_pos_ref[0]
            load_orientation_ref = load_orientation_ref[0]
            load_vel_ref = load_vel_ref[0]
            load_ang_vel_ref = load_ang_vel_ref[0]

        # to plot ref and actual pos side by side
        both_load_pos = torch.cat((load_pos_ref, load_pos), dim=-1)
        both_load_orientation = torch.cat((load_orientation_ref, load_orientation), dim=-1)
        both_load_vel = torch.cat((load_vel_ref, load_vel), dim=-1)
        both_load_ang_vel = torch.cat((load_ang_vel_ref, load_ang_vel), dim=-1)

        if not self.load_data:
            self.load_data = {
                "both_load_pos": both_load_pos.unsqueeze(0).tolist(),
                "both_load_orientation": both_load_orientation.unsqueeze(0).tolist(),
                "both_load_vel": both_load_vel.unsqueeze(0).tolist(),
                "both_load_ang_vel": both_load_ang_vel.unsqueeze(0).tolist(),
                "load_acc": load_acc.unsqueeze(0).tolist(),
                "load_ang_acc": load_ang_acc.unsqueeze(0).tolist()
            }
        else:
            self.load_data["both_load_pos"].append(both_load_pos.tolist())
            self.load_data["both_load_orientation"].append(both_load_orientation.tolist())
            self.load_data["both_load_vel"].append(both_load_vel.tolist())
            self.load_data["both_load_ang_vel"].append(both_load_ang_vel.tolist())
            self.load_data["load_acc"].append(load_acc.tolist())
            self.load_data["load_ang_acc"].append(load_ang_acc.tolist())

    def collect_drone_data(self):
        """Collect the drone data from the environment."""
        drone_pos = self.robot.data.body_state_w[:, self.drone_idx, :3][0]
        drone_orientation = self.robot.data.body_state_w[:, self.drone_idx, 3:7][0]
        drone_vel = self.robot.data.body_state_w[:, self.drone_idx, 7:10][0]
        drone_ang_vel = self.robot.data.body_state_w[:, self.drone_idx, 10:][0]
        drone_acc = self.robot.data.body_state_w[:, self.drone_idx, 10:][0]
        drone_ang_acc = self.robot.data.body_state_w[:, self.drone_idx, 10:][0]
        drone_jerk = self.env.action_manager._terms["low_level_action"]._drone_jerk[0]
        rotor_forces = self.env.action_manager._terms["low_level_action"].processed_actions[0][..., 2] # 3 * 4 rotors
        policy_ref = self.env.action_manager._terms["low_level_action"].raw_actions[0]
        action_space = policy_ref.shape[-1]/3 # 3 drones and every output has 3 dimensions        
        # Initialize a dictionary to store data for each drone
        if not hasattr(self, "drone_data_by_id"):
            self.drone_data_by_id = {}

        # Loop through all drones
        for drone_num in range(drone_pos.shape[0]):
            ref_drone = policy_ref[drone_num * int(action_space) : (drone_num + 1) * int(action_space)]
            ref_pos = ref_drone[:3]
            ref_vel = ref_drone[3:6]
            ref_acc = ref_drone[6:9]
            ref_jerk = ref_drone[9:12]
            # Append the data for this drone
            both_drone_pos = torch.cat((ref_pos, drone_pos[drone_num]), dim=-1)
            both_drone_vel = torch.cat((ref_vel, drone_vel[drone_num]), dim=-1)
            both_drone_acc = torch.cat((ref_acc, drone_acc[drone_num]), dim=-1)
            both_drone_jerk = torch.cat((ref_jerk, drone_jerk[drone_num]), dim=-1)
            # If this drone's data doesn't exist yet, initialize it
            if drone_num not in self.drone_data_by_id:
                self.drone_data_by_id[drone_num] = {
                    "both_drone_pos": both_drone_pos.unsqueeze(0).tolist(),
                    "drone_orientation": drone_orientation[drone_num].unsqueeze(0).tolist(),
                    "both_drone_vel": both_drone_vel.unsqueeze(0).tolist(),
                    "drone_ang_vel": drone_ang_vel[drone_num].unsqueeze(0).tolist(),
                    "both_drone_acc": both_drone_acc.unsqueeze(0).tolist(),
                    "drone_ang_acc": drone_ang_acc[drone_num].unsqueeze(0).tolist(),
                    "both_drone_jerk": both_drone_jerk.unsqueeze(0).tolist(),
                    "rotor_forces": rotor_forces[(drone_num * 4): (drone_num * 4) + 4].unsqueeze(0).tolist()
                }
            else:
                # Append the data for this drone
                self.drone_data_by_id[drone_num]["both_drone_pos"].append(both_drone_pos.tolist())
                self.drone_data_by_id[drone_num]["drone_orientation"].append(drone_orientation[drone_num].tolist())
                self.drone_data_by_id[drone_num]["both_drone_vel"].append(both_drone_vel.tolist())
                self.drone_data_by_id[drone_num]["drone_ang_vel"].append(drone_ang_vel[drone_num].tolist())
                self.drone_data_by_id[drone_num]["both_drone_acc"].append(both_drone_acc.tolist())
                self.drone_data_by_id[drone_num]["drone_ang_acc"].append(drone_ang_acc[drone_num].tolist())
                self.drone_data_by_id[drone_num]["both_drone_jerk"].append(both_drone_jerk.tolist())
                self.drone_data_by_id[drone_num]["rotor_forces"].append(rotor_forces[(drone_num * 4): (drone_num * 4) + 4].tolist())

    def collect_data(self):
        """Collect all the data in the environment."""
        self.collect_metrics()
        self.collect_load_data()
        self.collect_drone_data()


    def plot(self):
        """Plot the data."""
        all_data = {
        **{key: self.metrics[key] for key in self.metrics},
        **{key: self.load_data[key] for key in self.load_data},
        **{f"{key} Drone {drone_num}": self.drone_data_by_id[drone_num][key]
           for drone_num in self.drone_data_by_id
           for key in self.drone_data_by_id[drone_num]}
        }
        
        # Determine the number of subplots needed
        num_plots = len(all_data)
        plots_per_figure = 6
        num_figures = math.ceil(num_plots / plots_per_figure)

        # Create subplots
        keys = list(all_data.keys())
        for fig_idx in range(num_figures):
            plt.figure(figsize=(15, 10))  # Adjust figure size as needed
            start_idx = fig_idx * plots_per_figure
            end_idx = min(start_idx + plots_per_figure, num_plots)
            
            for subplot_idx, key_idx in enumerate(range(start_idx, end_idx)):
                key = keys[key_idx]
                ax = plt.subplot(2, 3, subplot_idx + 1)  # 2 rows, 3 columns
                
                # Plot data
                if "both" in key:
                    if "orientation" in key:
                        ref_data = [entry[:4] for entry in all_data[key]]
                        actual_data = [entry[4:] for entry in all_data[key]]
                        # Define your colors
                        colors = ['red', 'green', 'blue', 'purple']

                        # Plot each entry in ref_data with a different color
                        for i, color in enumerate(colors):
                            ax.plot(
                                [j * self.sim_dt for j in range(len(ref_data))], 
                                [data[i] for data in ref_data],  # Extract the i-th entry from each sublist
                                linestyle="--",
                                color=color,
                            )
                            ax.plot(
                                [j * self.sim_dt for j in range(len(actual_data))], 
                                [data[i] for data in actual_data],  # Extract the i-th entry from each sublist
                                color=color,
                            )
                        ax.legend(['W_ref', 'W', 'X_ref', 'X', 'Y_ref', 'Y', 'Z_ref', 'Z'])

                    else:
                        # Plot each entry in ref_data with a different color
                        ref_data = [entry[:3] for entry in all_data[key]]
                        actual_data = [entry[3:] for entry in all_data[key]]
                        colors = ['red', 'green', 'blue']
                        for i, color in enumerate(colors):
                            ax.plot(
                                [j * self.sim_dt for j in range(len(ref_data))], 
                                [data[i] for data in ref_data],  # Extract the i-th entry from each sublist
                                linestyle="--",
                                color=color,
                            )
                            ax.plot(
                                [j * self.sim_dt for j in range(len(actual_data))], 
                                [data[i] for data in actual_data],  # Extract the i-th entry from each sublist
                                color=color,
                            )
                        ax.legend(['X_ref', 'X', 'Y_ref', 'Y', 'Z_ref', 'Z'])
                else:
                    if "error" in key:
                        ax.plot(
                            [j * self.sim_dt for j in range(len(all_data[key]))], 
                            all_data[key], color = 'red'
                        )
                        ax.legend(['Norm error'])
                    else:
                        ax.plot(
                                [j * self.sim_dt for j in range(len(all_data[key]))], 
                                all_data[key]
                            )
                        if "orientation" in key:
                            ax.legend(['W', 'X', 'Y', 'Z'])
                        elif "rotor_forces" in key:
                            ax.legend(['Rotor 1', 'Rotor 2', 'Rotor 3', 'Rotor 4'])
                        else:
                            ax.legend(['X', 'Y', 'Z'])
                    
                ax.set_title(key)
                ax.set_xlabel("Time")
                ax.set_ylabel(key.split(" ")[0])  # Short label for Y-axis
            
        plt.tight_layout()  # Adjust layout to avoid overlap
        plt.show()

    def save(self):
        """Save the data."""
        pass
