"""Helper class to plot results of ManagerBasedRLEnv"""

import math
import matplotlib.pyplot as plt
import os
import torch

from isaaclab.envs import DirectMARLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate


class DirectMARLPlotter:
    def __init__(self, env: DirectMARLEnv, control_mode: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):

        # environment
        self.control_mode = control_mode
        self.env = env
        self.robot = env.scene[asset_cfg.name]
        self.load_id = self.robot.find_bodies("load_odometry_sensor_link")[0]
        self.drone_idx = self.robot.find_bodies("Falcon.*base_link")[0]
        self.sim_dt = env.sim.get_rendering_dt()

        # buffers
        self.metrics: dict = {}
        self.load_data: dict = {}
        self.drone_data_by_id: dict = {}

    def collect_metrics(self):
        """Collect the metrics from the environment."""
        if not self.metrics:
            self.metrics = self.env.metrics.copy()
            for key in self.metrics:
                self.metrics[key] = self.metrics[key].tolist()
        else:
            for key in self.metrics:
                self.metrics[key].append(self.env.metrics[key].item())

    def collect_load_data(self):
        """Collect the load data from the environment."""
        # load data
        load_pos = self.robot.data.body_com_state_w[:, self.load_id, :3].squeeze(1)[0]
        load_orientation = self.robot.data.body_com_state_w[:, self.load_id, 3:7].squeeze(1)[0]
        load_vel = self.robot.data.body_com_state_w[:, self.load_id, 7:10].squeeze(1)[0]
        load_ang_vel = self.robot.data.body_com_state_w[:, self.load_id, 10:].squeeze(1)[0]
        load_acc = self.robot.data.body_com_state_w[:, self.load_id, 10:].squeeze(1)[0]
        load_ang_acc = self.robot.data.body_com_state_w[:, self.load_id, 10:].squeeze(1)[0]

        # references
        load_pos_ref = self.env.pose_command_w[:, :3][0]
        load_orientation_ref = self.env.pose_command_w[..., 3:7][0]

        # to plot ref and actual pos side by side
        both_load_pos = torch.cat((load_pos_ref, load_pos), dim=-1)
        both_load_orientation = torch.cat((load_orientation_ref, load_orientation), dim=-1)

        if not self.load_data:
            self.load_data = {
                "both_load_pos": both_load_pos.unsqueeze(0).tolist(),
                "both_load_orientation": both_load_orientation.unsqueeze(0).tolist(),
                "load_vel": load_vel.unsqueeze(0).tolist(),
                "load_ang_vel": load_ang_vel.unsqueeze(0).tolist(),
                "load_acc": load_acc.unsqueeze(0).tolist(),
                "load_ang_acc": load_ang_acc.unsqueeze(0).tolist(),
            }
        else:
            self.load_data["both_load_pos"].append(both_load_pos.tolist())
            self.load_data["both_load_orientation"].append(both_load_orientation.tolist())
            self.load_data["load_vel"].append(load_vel.tolist())
            self.load_data["load_ang_vel"].append(load_ang_vel.tolist())
            self.load_data["load_acc"].append(load_acc.tolist())
            self.load_data["load_ang_acc"].append(load_ang_acc.tolist())

    def collect_drone_data(self):
        """Collect the drone data from the environment."""
        drone_pos = self.robot.data.body_com_state_w[:, self.drone_idx, :3][0]
        drone_orientation = self.robot.data.body_com_state_w[:, self.drone_idx, 3:7][0]
        drone_vel = self.robot.data.body_com_state_w[:, self.drone_idx, 7:10][0]
        drone_ang_vel = self.robot.data.body_com_state_w[:, self.drone_idx, 10:][0]
        drone_BR = quat_rotate(drone_orientation.unsqueeze(0), drone_ang_vel.unsqueeze(0))[0]
        drone_acc = self.robot.data.body_acc_w[:, self.drone_idx, :3][0]
        drone_ang_acc = self.robot.data.body_acc_w[:, self.drone_idx, 3:6][0]
        drone_jerk = self.env._drone_jerk[0]
        rotor_forces = self.env._forces[0][..., 2]  # 3 * 4 rotors
        policy_refs = self.env.actions
        policy_ref = torch.cat([action[0] for drone_num, action in policy_refs.items()])
        action_space = policy_ref.shape[-1] / 3  # 3 drones and every output has 3 dimensions
        # Initialize a dictionary to store data for each drone
        if not hasattr(self, "drone_data_by_id"):
            self.drone_data_by_id = {}

        # Loop through all drones
        for drone_num in range(drone_pos.shape[0]):
            ref_drone = policy_ref[drone_num * int(action_space) : (drone_num + 1) * int(action_space)]
            filtered_acc = self.env.geo_controllers[drone_num].filtered_acc[0]
            filtered_rate = self.env.geo_controllers[drone_num].filtered_rate[0]
            unfiltered_thrusts_geo = self.env.geo_controllers[drone_num].unfiltered_thrusts[0]
            filtered_thrusts_geo = self.env.geo_controllers[drone_num].filtered_thrusts[0]

            unfiltered_mot = self.env._indi_controllers[drone_num].unfiltered_mot[0]
            filtered_mot = self.env._indi_controllers[drone_num].filtered_mot[0]
            filtered_ang_acc = self.env._indi_controllers[drone_num].filtered_ang_acc[0]

            if self.control_mode == "ACCBR":
                ref_acc = ref_drone[:3]
                ref_BR = torch.cat((ref_drone[3:], torch.zeros((1), device="cuda")), dim=-1)
                # Append the data for this drone

                both_drone_acc = torch.cat((ref_acc, drone_acc[drone_num]), dim=-1)
                both_drone_BR = torch.cat((ref_BR, drone_BR[drone_num]), dim=-1)

                both_filter_acc = torch.cat((drone_acc[drone_num], filtered_acc), dim=-1)
                both_filter_rate = torch.cat((drone_ang_vel[drone_num], filtered_rate), dim=-1)
                both_filter_cthrust = torch.cat((unfiltered_thrusts_geo, filtered_thrusts_geo), dim=-1)

                both_filter_ang_acc = torch.cat((drone_ang_acc[drone_num], filtered_ang_acc), dim=-1)
                both_filter_mot = torch.cat((unfiltered_mot, filtered_mot), dim=-1)

                # If this drone's data doesn't exist yet, initialize it
                if drone_num not in self.drone_data_by_id:
                    self.drone_data_by_id[drone_num] = {
                        "drone_pos": drone_pos[drone_num].unsqueeze(0).tolist(),
                        "drone_orientation": drone_orientation[drone_num].unsqueeze(0).tolist(),
                        "drone_vel": drone_vel[drone_num].unsqueeze(0).tolist(),
                        "drone_ang_vel": drone_ang_vel[drone_num].unsqueeze(0).tolist(),
                        "both_filter_rate_geo": both_filter_rate.unsqueeze(0).tolist(),
                        "both_drone_BR": both_drone_BR.unsqueeze(0).tolist(),
                        "both_drone_acc": both_drone_acc.unsqueeze(0).tolist(),
                        "both_filter_acc_geo": both_filter_acc.unsqueeze(0).tolist(),
                        "both_filter_ang_acc_indi": both_filter_ang_acc.unsqueeze(0).tolist(),
                        "drone_jerk": drone_jerk[drone_num].unsqueeze(0).tolist(),
                        "both_filter_cthrust_geo": both_filter_cthrust.unsqueeze(0).tolist(),
                        "both_filter_mot_indi": both_filter_mot.unsqueeze(0).tolist(),
                        "rotor_forces": rotor_forces[(drone_num * 4) : (drone_num * 4) + 4].unsqueeze(0).tolist(),
                    }
                else:
                    # Append the data for this drone
                    self.drone_data_by_id[drone_num]["drone_pos"].append(drone_pos[drone_num].tolist())
                    self.drone_data_by_id[drone_num]["drone_orientation"].append(drone_orientation[drone_num].tolist())
                    self.drone_data_by_id[drone_num]["drone_vel"].append(drone_vel[drone_num].tolist())
                    self.drone_data_by_id[drone_num]["drone_ang_vel"].append(drone_ang_vel[drone_num].tolist())
                    self.drone_data_by_id[drone_num]["both_filter_rate_geo"].append(both_filter_rate.tolist())
                    self.drone_data_by_id[drone_num]["both_drone_BR"].append(both_drone_BR.tolist())
                    self.drone_data_by_id[drone_num]["both_drone_acc"].append(both_drone_acc.tolist())
                    self.drone_data_by_id[drone_num]["both_filter_acc_geo"].append(both_filter_acc.tolist())
                    self.drone_data_by_id[drone_num]["both_filter_ang_acc_indi"].append(both_filter_ang_acc.tolist())
                    self.drone_data_by_id[drone_num]["drone_jerk"].append(drone_jerk[drone_num].tolist())
                    self.drone_data_by_id[drone_num]["both_filter_cthrust_geo"].append(both_filter_cthrust.tolist())
                    self.drone_data_by_id[drone_num]["both_filter_mot_indi"].append(both_filter_mot.tolist())
                    self.drone_data_by_id[drone_num]["rotor_forces"].append(
                        rotor_forces[(drone_num * 4) : (drone_num * 4) + 4].tolist()
                    )

            elif self.control_mode == "geometric":
                ref_pos = ref_drone[:3]
                ref_vel = ref_drone[3:6]
                ref_acc = ref_drone[6:9]
                ref_jerk = ref_drone[9:12]
                # Append the data for this drone
                both_drone_pos = torch.cat((ref_pos, drone_pos[drone_num]), dim=-1)
                both_drone_vel = torch.cat((ref_vel, drone_vel[drone_num]), dim=-1)
                both_drone_acc = torch.cat((ref_acc, drone_acc[drone_num]), dim=-1)
                both_drone_jerk = torch.cat((ref_jerk, drone_jerk[drone_num]), dim=-1)

                both_filter_acc = torch.cat((drone_acc[drone_num], filtered_acc), dim=-1)
                both_filter_rate = torch.cat((drone_ang_vel[drone_num], filtered_rate), dim=-1)
                both_filter_cthrust = torch.cat((unfiltered_thrusts_geo, filtered_thrusts_geo), dim=-1)

                both_filter_ang_acc = torch.cat((drone_ang_acc[drone_num], filtered_ang_acc), dim=-1)
                both_filter_mot = torch.cat((unfiltered_mot, filtered_mot), dim=-1)

                # If this drone's data doesn't exist yet, initialize it
                if drone_num not in self.drone_data_by_id:
                    self.drone_data_by_id[drone_num] = {
                        "both_drone_pos": both_drone_pos.unsqueeze(0).tolist(),
                        "drone_orientation": drone_orientation[drone_num].unsqueeze(0).tolist(),
                        "both_drone_vel": both_drone_vel.unsqueeze(0).tolist(),
                        "drone_ang_vel": drone_ang_vel[drone_num].unsqueeze(0).tolist(),
                        "both_filter_rate_geo": both_filter_rate.unsqueeze(0).tolist(),
                        "both_drone_acc": both_drone_acc.unsqueeze(0).tolist(),
                        "both_filter_acc_geo": both_filter_acc.unsqueeze(0).tolist(),
                        "both_filter_ang_acc_indi": both_filter_ang_acc.unsqueeze(0).tolist(),
                        "both_drone_jerk": both_drone_jerk.unsqueeze(0).tolist(),
                        "both_filter_cthrust_geo": both_filter_cthrust.unsqueeze(0).tolist(),
                        "both_filter_mot_indi": both_filter_mot.unsqueeze(0).tolist(),
                        "rotor_forces": rotor_forces[(drone_num * 4) : (drone_num * 4) + 4].unsqueeze(0).tolist(),
                    }
                else:
                    # Append the data for this drone
                    self.drone_data_by_id[drone_num]["both_drone_pos"].append(both_drone_pos.tolist())
                    self.drone_data_by_id[drone_num]["drone_orientation"].append(drone_orientation[drone_num].tolist())
                    self.drone_data_by_id[drone_num]["both_drone_vel"].append(both_drone_vel.tolist())
                    self.drone_data_by_id[drone_num]["drone_ang_vel"].append(drone_ang_vel[drone_num].tolist())
                    self.drone_data_by_id[drone_num]["both_filter_rate_geo"].append(both_filter_rate.tolist())
                    self.drone_data_by_id[drone_num]["both_drone_acc"].append(both_drone_acc.tolist())
                    self.drone_data_by_id[drone_num]["both_filter_acc_geo"].append(both_filter_acc.tolist())
                    self.drone_data_by_id[drone_num]["both_filter_ang_acc_indi"].append(both_filter_ang_acc.tolist())
                    self.drone_data_by_id[drone_num]["both_drone_jerk"].append(both_drone_jerk.tolist())
                    self.drone_data_by_id[drone_num]["both_filter_cthrust_geo"].append(both_filter_cthrust.tolist())
                    self.drone_data_by_id[drone_num]["both_filter_mot_indi"].append(both_filter_mot.tolist())
                    self.drone_data_by_id[drone_num]["rotor_forces"].append(
                        rotor_forces[(drone_num * 4) : (drone_num * 4) + 4].tolist()
                    )

    def collect_data(self):
        """Collect all the data in the environment."""
        self.collect_metrics()
        self.collect_load_data()
        self.collect_drone_data()

    def plot(self, save=False, save_dir="plots", file_format="png"):
        """Plot the data and optionally save the plots to files."""
        # Consolidate all data into one dictionary
        all_data = {
            **self.metrics,
            **self.load_data,
            **{
                f"{key} Drone {drone_num}": self.drone_data_by_id[drone_num][key]
                for drone_num in self.drone_data_by_id
                for key in self.drone_data_by_id[drone_num]
            },
        }

        # Determine the number of subplots needed
        num_plots = len(all_data)
        plots_per_figure = 6
        num_figures = math.ceil(num_plots / plots_per_figure)

        # Define reusable plotting logic
        def plot_entries(ax, time_data, data, colors, linestyle, labels=None):
            for i, color in enumerate(colors):
                ax.plot(
                    time_data,
                    [entry[i] for entry in data],
                    linestyle=linestyle,
                    color=color,
                )
            if labels:
                ax.legend(labels)

        # Ensure the save directory exists if saving
        if save:
            os.makedirs(save_dir, exist_ok=True)

        # Plot all data
        keys = list(all_data.keys())
        time_data = [i * self.sim_dt for i in range(len(next(iter(all_data.values()))))]

        for fig_idx in range(num_figures):
            fig = plt.figure(figsize=(15, 10))
            start_idx = fig_idx * plots_per_figure
            end_idx = min(start_idx + plots_per_figure, num_plots)

            for subplot_idx, key_idx in enumerate(range(start_idx, end_idx)):
                key = keys[key_idx]
                ax = plt.subplot(2, 3, subplot_idx + 1)
                data = all_data[key]

                if "both" in key:
                    if "orientation" in key or "mot" in key:
                        ref_data = [entry[:4] for entry in data]
                        actual_data = [entry[4:] for entry in data]
                        colors = ["red", "green", "blue", "purple"]
                        plot_entries(ax, time_data, ref_data, colors, linestyle="--")
                        print("key", key)
                        plot_entries(ax, time_data, actual_data, colors, linestyle="-")
                        if "orientation" in key:
                            ax.legend(["W_ref", "X_ref", "Y_ref", "Z_ref", "W", "X", "Y", "Z"])
                        else:
                            ax.legend(
                                ["F1", "F2", "F3", "F4", "F1_filtered", "F2_filtered", "F3_filtered", "F4_filtered"]
                            )
                    else:
                        ref_data = [entry[:3] for entry in data]
                        actual_data = [entry[3:] for entry in data]
                        colors = ["red", "green", "blue"]
                        plot_entries(ax, time_data, ref_data, colors, linestyle="--")
                        plot_entries(ax, time_data, actual_data, colors, linestyle="-")
                        ax.legend(["X_ref", "Y_ref", "Z_ref", "X", "Y", "Z"])
                else:
                    if "error" in key:
                        ax.plot(time_data, data, color="red")
                        ax.legend(["Norm error"])
                    else:
                        ax.plot(time_data, data)
                        if "orientation" in key:
                            ax.legend(["W", "X", "Y", "Z"])
                        elif "rotor_forces" in key:
                            ax.legend(["Rotor 1", "Rotor 2", "Rotor 3", "Rotor 4"])
                        else:
                            ax.legend(["X", "Y", "Z"])

                ax.set_title(key)
                ax.set_xlabel("Time")
                ax.set_ylabel(key.split(" ")[0])

            # Save the figure if requested
            if save:
                save_path = os.path.join(save_dir, f"figure_{fig_idx + 1}.{file_format}")
                fig.savefig(save_path)
                print(f"Figure saved: {save_path}")

        plt.tight_layout()
        plt.show()
