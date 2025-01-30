import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def quintic_trajectory_coeffs(waypoints: torch.tensor, 
                       time_horizon: float, 
                       num_envs: int) -> tuple:
    # T0 is 0
    T1 = 0.2 * time_horizon
    T2 = 0.4 * time_horizon
    T3 = 0.6 * time_horizon
    T4 = 0.8 * time_horizon
    T5 = time_horizon

    A = torch.tensor(
        [
            [0, 0, 0, 0, 0, 1],  # p(t0)
            [T1**5, T1**4, T1**3, T1**2, T1, 1],  # p(t1)
            [T2**5, T2**4, T2**3, T2**2, T2, 1],  # p(t2)
            [T3**5, T3**4, T3**3, T3**2, T3, 1],  # p(t3)
            [T4**5, T4**4, T4**3, T4**2, T4, 1],  # p(t4)
            [T5**5, T5**4, T5**3, T5**2, T5, 1],  # p(t5)
        ],
        dtype=torch.float32,
    ).repeat(num_envs, 1, 1)

    # Position polynomials
    b_x = (
        torch.stack(
            [
                waypoints[:, 0][:, 0],
                waypoints[:, 1][:, 0],
                waypoints[:, 2][:, 0],
                waypoints[:, 3][:, 0],
                waypoints[:, 4][:, 0],
                waypoints[:, 5][:, 0],
            ],
            dim=1,
        )
        .float()
    )
    b_y = (
        torch.stack(
            [
                waypoints[:, 0][:, 1],
                waypoints[:, 1][:, 1],
                waypoints[:, 2][:, 1],
                waypoints[:, 3][:, 1],
                waypoints[:, 4][:, 1],
                waypoints[:, 5][:, 1],
            ],
            dim=1,
        )
        .float()
    )
    b_z = (
        torch.stack(
            [
                waypoints[:, 0][:, 2],
                waypoints[:, 1][:, 2],
                waypoints[:, 2][:, 2],
                waypoints[:, 3][:, 2],
                waypoints[:, 4][:, 2],
                waypoints[:, 5][:, 2],
            ],
            dim=1,
        )
        .float()
    )

    coeffs_x = torch.linalg.solve(A, b_x)
    coeffs_y = torch.linalg.solve(A, b_y)
    coeffs_z = torch.linalg.solve(A, b_z)

    coeffs = torch.stack([coeffs_x, coeffs_y, coeffs_z], dim=1)

    return coeffs


# Compute derivatives of the polynomial (velocity, acceleration, jerk, snap)
def compute_derivatives(coeffs, power):
    if power == 1:
        return coeffs[:, :-1] * torch.tensor([5, 4, 3, 2, 1], dtype=torch.float32)
    elif power == 2:
        return coeffs[:, :-2] * torch.tensor([20, 12, 6, 2], dtype=torch.float32)
    elif power == 3:
        return coeffs[:, :-3] * torch.tensor([60, 24, 6], dtype=torch.float32)
    elif power == 4:
        return coeffs[:, :-4] * torch.tensor([120, 24], dtype=torch.float32)
    else:
        return coeffs

# Compute trajectory at time t_eval
def evaluate_trajectory(coeffs, time_eval, time_horizon):
    
    T = time_eval * time_horizon

    coeffs_x = coeffs[:, 0]
    coeffs_y = coeffs[:, 1]
    coeffs_z = coeffs[:, 2]

    # Compute position (5th order polynomial)
    position_x = torch.sum(
        coeffs_x
        * torch.tensor([T**5, T**4, T**3, T**2, T, 1], dtype=torch.float32),
        dim=-1,
        keepdim=True,
    )
    position_y = torch.sum(
        coeffs_y
        * torch.tensor([T**5, T**4, T**3, T**2, T, 1], dtype=torch.float32),
        dim=-1,
        keepdim=True,
    )
    position_z = torch.sum(
        coeffs_z
        * torch.tensor([T**5, T**4, T**3, T**2, T, 1], dtype=torch.float32),
        dim=-1,
        keepdim=True,
    )
    position = torch.cat([position_x, position_y, position_z], dim=-1)

    # Compute velocity
    velocity_x = torch.sum(
        compute_derivatives(coeffs_x, 1)
        * torch.tensor([T**4, T**3, T**2, T, 1], dtype=torch.float32),
        dim=-1,
        keepdim=True,
    )
    velocity_y = torch.sum(
        compute_derivatives(coeffs_y, 1)
        * torch.tensor([T**4, T**3, T**2, T, 1], dtype=torch.float32),
        dim=-1,
        keepdim=True,
    )
    velocity_z = torch.sum(
        compute_derivatives(coeffs_z, 1)
        * torch.tensor([T**4, T**3, T**2, T, 1], dtype=torch.float32),
        dim=-1,
        keepdim=True,
    )
    velocity = torch.cat([velocity_x, velocity_y, velocity_z], dim=-1)

    # Compute acceleration
    acceleration_x = torch.sum(
        compute_derivatives(coeffs_x, 2)
        * torch.tensor([T**3, T**2, T, 1], dtype=torch.float32),
        dim=-1,
        keepdim=True,
    )
    acceleration_y = torch.sum(
        compute_derivatives(coeffs_y, 2)
        * torch.tensor([T**3, T**2, T, 1], dtype=torch.float32),
        dim=-1,
        keepdim=True,
    )
    acceleration_z = torch.sum(
        compute_derivatives(coeffs_z, 2)
        * torch.tensor([T**3, T**2, T, 1], dtype=torch.float32),
        dim=-1,
        keepdim=True,
    )
    acceleration = torch.cat([acceleration_x, acceleration_y, acceleration_z], dim=-1)

    # Compute jerk
    jerk_x = torch.sum(
        compute_derivatives(coeffs_x, 3)
        * torch.tensor([T**2, T, 1], dtype=torch.float32),
        dim=-1,
        keepdim=True,
    )
    jerk_y = torch.sum(
        compute_derivatives(coeffs_y, 3)
        * torch.tensor([T**2, T, 1], dtype=torch.float32),
        dim=-1,
        keepdim=True,
    )
    jerk_z = torch.sum(
        compute_derivatives(coeffs_z, 3)
        * torch.tensor([T**2, T, 1], dtype=torch.float32),
        dim=-1,
        keepdim=True,
    )
    jerk = torch.cat([jerk_x, jerk_y, jerk_z], dim=-1)

    # Compute snap
    # snap_x = torch.sum(
    #     compute_derivatives(coeffs_x, 4)
    #     * torch.tensor([T, 1], dtype=torch.float32),
    #     dim=-1,
    #     keepdim=True,
    # )
    # snap_y = torch.sum(
    #     compute_derivatives(coeffs_y, 4)
    #     * torch.tensor([T, 1], dtype=torch.float32),
    #     dim=-1,
    #     keepdim=True,
    # )
    # snap_z = torch.sum(
    #     compute_derivatives(coeffs_z, 4)
    #     * torch.tensor([T, 1], dtype=torch.float32),
    #     dim=-1,
    #     keepdim=True,
    # )
    # snap = torch.cat([snap_x, snap_y, snap_z], dim=-1)

    return position, velocity, acceleration, jerk #, snap

# helper function to update buffer

def update_buffer(buffer, drone_id, new_position):
    """
    Updates the buffer with the new position, moving older entries forward.
    
    Args:
        buffer (torch.Tensor): The buffer holding the last 6 positions (shape: [num_envs, num_drones, buffer_size, 3]).
        drone_id (int): The ID of the drone to update.
        new_position (torch.Tensor): The current position of the drone (shape: [num_envs, 3]).
    
    Returns:
        torch.Tensor: The updated buffer.
    """
    # Clone the slice to avoid in-place memory conflict
    buffer[:, drone_id, :-1] = buffer[:, drone_id, 1:].clone()
    
    # Add the new position at the end (back) of the buffer
    buffer[:, drone_id, -1] = new_position
    
    return buffer


spline_positions = torch.zeros(1, 3, 6, 3)
spline_coeffs = torch.zeros(1, 3, 3, 6)
spline_time_horizon = 10.0

# set postions for drone 1
spline_positions[:, 0, 0, :] = torch.tensor([0.0, 0.0, 0.0])
spline_positions[:, 0, 1, :] = torch.tensor([1.0, 1.0, 1.0])
spline_positions[:, 0, 2, :] = torch.tensor([1.5, 1.0, 1.0])
spline_positions[:, 0, 3, :] = torch.tensor([1.5, 1.5, 1.0])
spline_positions[:, 0, 4, :] = torch.tensor([1.5, 1.5, 1.5])
spline_positions[:, 0, 5, :] = torch.tensor([2.0, 2.0, 2.0])

# set postions for drone 2
spline_positions[:, 1, 0, :] = torch.tensor([0.0, 0.0, 0.0])
spline_positions[:, 1, 1, :] = torch.tensor([-1.0, -1.0, 1.0])
spline_positions[:, 1, 2, :] = torch.tensor([-1.5, -1.0, 1.0])
spline_positions[:, 1, 3, :] = torch.tensor([-1.5, -1.5, 1.0])
spline_positions[:, 1, 4, :] = torch.tensor([-1.5, -1.5, 1.5])
spline_positions[:, 1, 5, :] = torch.tensor([-2.0, -2.0, 2.0])

# set postions for drone 3
spline_positions[:, 2, 0, :] = torch.tensor([0.0, 0.0, 0.0])
spline_positions[:, 2, 1, :] = torch.tensor([1.0, -1.0, 1.0])
spline_positions[:, 2, 2, :] = torch.tensor([1.5, -1.0, 1.0])
spline_positions[:, 2, 3, :] = torch.tensor([1.5, -1.5, 1.0])
spline_positions[:, 2, 4, :] = torch.tensor([1.5, -1.5, 1.5])
spline_positions[:, 2, 5, :] = torch.tensor([2.0, -2.0, 2.0])

# compute spline coefficients
spline_coeffs[:, 0] = quintic_trajectory_coeffs(spline_positions[:, 0], spline_time_horizon, 1)
spline_coeffs[:, 1] = quintic_trajectory_coeffs(spline_positions[:, 1], spline_time_horizon, 1)
spline_coeffs[:, 2] = quintic_trajectory_coeffs(spline_positions[:, 2], spline_time_horizon, 1)

# Generate time samples
num_samples = 100
time_samples = np.linspace(0, 1, num_samples)

# Evaluate the trajectories
trajectories = {0: [], 1: [], 2: []}
velocities = {0: [], 1: [], 2: []}
accelerations = {0: [], 1: [], 2: []}
jerks = {0: [], 1: [], 2: []}

for t in time_samples:
    for drone_id in range(3):
        pos, vel, acc, jerk = evaluate_trajectory(spline_coeffs[:, drone_id], t, spline_time_horizon)
        trajectories[drone_id].append(pos)
        velocities[drone_id].append(vel)
        accelerations[drone_id].append(acc)
        jerks[drone_id].append(jerk)

# Convert lists to numpy arrays for plotting
for key in trajectories:
    trajectories[key] = np.array(trajectories[key])
    velocities[key] = np.array(velocities[key])
    accelerations[key] = np.array(accelerations[key])
    jerks[key] = np.array(jerks[key])

# Plot the trajectories
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Define colors for each drone
colors = ['r', 'g', 'b']
labels = ['Drone 1', 'Drone 2', 'Drone 3']

for drone_id in range(3):
    traj = trajectories[drone_id]
    ax.plot(traj[..., 0], traj[..., 1], traj[..., 2], color=colors[drone_id], label=labels[drone_id])

# Formatting the plot
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('3D Spline Trajectories of Drones')

fig, axes = plt.subplots(3, 1, figsize=(8, 12))
fig.suptitle("Drone Velocities")
for i in range(3):
    axes[i].plot(time_samples * spline_time_horizon, velocities[i].squeeze(), label=['X', 'Y', 'Z'])
    axes[i].set_title(f'Drone {i+1}')
    axes[i].legend()
    axes[i].set_xlabel('Time (s)')
    axes[i].set_ylabel('Velocity')
    axes[i].grid()

plt.tight_layout()

fig, axes = plt.subplots(3, 1, figsize=(8, 12))
fig.suptitle("Drone Accelerations")
for i in range(3):
    axes[i].plot(time_samples * spline_time_horizon, accelerations[i].squeeze(), label=['X', 'Y', 'Z'])
    axes[i].set_title(f'Drone {i+1}')
    axes[i].legend()
    axes[i].set_xlabel('Time (s)')
    axes[i].set_ylabel('Acceleration')
    axes[i].grid()

plt.tight_layout()

fig, axes = plt.subplots(3, 1, figsize=(8, 12))
fig.suptitle("Drone Jerks")
for i in range(3):
    axes[i].plot(time_samples * spline_time_horizon, jerks[i].squeeze(), label=['X', 'Y', 'Z'])
    axes[i].set_title(f'Drone {i+1}')
    axes[i].legend()
    axes[i].set_xlabel('Time (s)')
    axes[i].set_ylabel('Jerk')
    axes[i].grid()

plt.tight_layout()
plt.show()

