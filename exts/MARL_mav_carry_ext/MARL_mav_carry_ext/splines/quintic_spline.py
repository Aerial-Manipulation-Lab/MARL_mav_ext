import torch


# Quintic polynomial trajectory for a single segment
def quintic_trajectory_coeffs(waypoints: torch.tensor, 
                       time_horizon: float, 
                       num_envs: int) -> tuple:
    # T0 is 0
    T1 = 0.5 * time_horizon
    T2 = time_horizon

    A = torch.tensor(
        [
            [0, 0, 0, 0, 0, 1],  # p(t0)
            [0, 0, 0, 0, 1, 0],  # v(t0)
            [T1**5, T1**4, T1**3, T1**2, T1, 1],  # p(t1)
            [5* T1**4, 4 * T1**3, 3 * T1**2, 2*T1, 1, 0],  # v(t1)
            [20 * T1**3, 12 * T1**2, 6 * T1, 2, 0, 0],  # a(t1)
            [T2**5, T2**4, T2**3, T2**2, T2, 1],  # p(t2)
        ],
        dtype=torch.float32,
        device=torch.device("cuda"),
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
        .to("cuda")
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
        .to("cuda")
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
        .to("cuda")
    )

    coeffs_x = torch.linalg.solve(A, b_x)
    coeffs_y = torch.linalg.solve(A, b_y)
    coeffs_z = torch.linalg.solve(A, b_z)

    coeffs = torch.stack([coeffs_x, coeffs_y, coeffs_z], dim=1)

    return coeffs


# Compute derivatives of the polynomial (velocity, acceleration, jerk, snap)
def compute_derivatives(coeffs, power):
    if power == 1:
        return coeffs[:, :-1] * torch.tensor([5, 4, 3, 2, 1], dtype=torch.float32, device=torch.device("cuda"))
    elif power == 2:
        return coeffs[:, :-2] * torch.tensor([20, 12, 6, 2], dtype=torch.float32, device=torch.device("cuda"))
    elif power == 3:
        return coeffs[:, :-3] * torch.tensor([60, 24, 6], dtype=torch.float32, device=torch.device("cuda"))
    elif power == 4:
        return coeffs[:, :-4] * torch.tensor([120, 24], dtype=torch.float32, device=torch.device("cuda"))
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
        * torch.tensor([T**5, T**4, T**3, T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")),
        dim=-1,
        keepdim=True,
    )
    position_y = torch.sum(
        coeffs_y
        * torch.tensor([T**5, T**4, T**3, T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")),
        dim=-1,
        keepdim=True,
    )
    position_z = torch.sum(
        coeffs_z
        * torch.tensor([T**5, T**4, T**3, T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")),
        dim=-1,
        keepdim=True,
    )
    position = torch.cat([position_x, position_y, position_z], dim=-1)

    # Compute velocity
    velocity_x = torch.sum(
        compute_derivatives(coeffs_x, 1)
        * torch.tensor([T**4, T**3, T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")),
        dim=-1,
        keepdim=True,
    )
    velocity_y = torch.sum(
        compute_derivatives(coeffs_y, 1)
        * torch.tensor([T**4, T**3, T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")),
        dim=-1,
        keepdim=True,
    )
    velocity_z = torch.sum(
        compute_derivatives(coeffs_z, 1)
        * torch.tensor([T**4, T**3, T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")),
        dim=-1,
        keepdim=True,
    )
    velocity = torch.cat([velocity_x, velocity_y, velocity_z], dim=-1)

    # Compute acceleration
    acceleration_x = torch.sum(
        compute_derivatives(coeffs_x, 2)
        * torch.tensor([T**3, T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")),
        dim=-1,
        keepdim=True,
    )
    acceleration_y = torch.sum(
        compute_derivatives(coeffs_y, 2)
        * torch.tensor([T**3, T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")),
        dim=-1,
        keepdim=True,
    )
    acceleration_z = torch.sum(
        compute_derivatives(coeffs_z, 2)
        * torch.tensor([T**3, T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")),
        dim=-1,
        keepdim=True,
    )
    acceleration = torch.cat([acceleration_x, acceleration_y, acceleration_z], dim=-1)

    # Compute jerk
    jerk_x = torch.sum(
        compute_derivatives(coeffs_x, 3)
        * torch.tensor([T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")),
        dim=-1,
        keepdim=True,
    )
    jerk_y = torch.sum(
        compute_derivatives(coeffs_y, 3)
        * torch.tensor([T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")),
        dim=-1,
        keepdim=True,
    )
    jerk_z = torch.sum(
        compute_derivatives(coeffs_z, 3)
        * torch.tensor([T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")),
        dim=-1,
        keepdim=True,
    )
    jerk = torch.cat([jerk_x, jerk_y, jerk_z], dim=-1)

    # Compute snap
    # snap_x = torch.sum(
    #     compute_derivatives(coeffs_x, 4)
    #     * torch.tensor([T, 1], dtype=torch.float32, device=torch.device("cuda")),
    #     dim=-1,
    #     keepdim=True,
    # )
    # snap_y = torch.sum(
    #     compute_derivatives(coeffs_y, 4)
    #     * torch.tensor([T, 1], dtype=torch.float32, device=torch.device("cuda")),
    #     dim=-1,
    #     keepdim=True,
    # )
    # snap_z = torch.sum(
    #     compute_derivatives(coeffs_z, 4)
    #     * torch.tensor([T, 1], dtype=torch.float32, device=torch.device("cuda")),
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
