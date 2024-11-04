import torch


# Septic polynomial trajectory for a single segment
def septic_trajectory(start, end, t_start, t_end, num_envs):
    T = t_end - t_start
    A = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 1],  # p(t0)
            [T**7, T**6, T**5, T**4, T**3, T**2, T, 1],  # p(t1)
            [0, 0, 0, 0, 0, 0, 1, 0],  # v(t0)
            [7 * T**6, 6 * T**5, 5 * T**4, 4 * T**3, 3 * T**2, 2 * T, 1, 0],  # v(t1)
            [0, 0, 0, 0, 0, 2, 0, 0],  # a(t0)
            [42 * T**5, 30 * T**4, 20 * T**3, 12 * T**2, 6 * T, 2, 0, 0],  # a(t1)
            [210 * T**4, 120 * T**3, 60 * T**2, 24 * T, 6, 0, 0, 0],  # j(t1)
            [840 * T**3, 360 * T**2, 120 * T, 24, 0, 0, 0, 0],  # s(t1)
        ],
        dtype=torch.float32,
        device=torch.device("cuda"),
    ).repeat(num_envs, 1, 1)

    # Position polynomials
    b_x = (
        torch.stack(
            [
                start["pos"][:, 0],
                end["pos"][:, 0],
                start["vel"][:, 0],
                end["vel"][:, 0],
                start["acc"][:, 0],
                end["acc"][:, 0],
                end["jerk"][:, 0],
                end["snap"][:, 0],
            ],
            dim=1,
        )
        .float()
        .to("cuda")
    )
    b_y = (
        torch.stack(
            [
                start["pos"][:, 1],
                end["pos"][:, 1],
                start["vel"][:, 1],
                end["vel"][:, 1],
                start["acc"][:, 1],
                end["acc"][:, 1],
                end["jerk"][:, 1],
                end["snap"][:, 1],
            ],
            dim=1,
        )
        .float()
        .to("cuda")
    )
    b_z = (
        torch.stack(
            [
                start["pos"][:, 2],
                end["pos"][:, 2],
                start["vel"][:, 2],
                end["vel"][:, 2],
                start["acc"][:, 2],
                end["acc"][:, 2],
                end["jerk"][:, 2],
                end["snap"][:, 2],
            ],
            dim=1,
        )
        .float()
        .to("cuda")
    )

    coeffs_x = torch.linalg.solve(A, b_x)
    coeffs_y = torch.linalg.solve(A, b_y)
    coeffs_z = torch.linalg.solve(A, b_z)

    return coeffs_x, coeffs_y, coeffs_z


# Compute derivatives of the polynomial (velocity, acceleration, jerk, snap)
def compute_derivatives(coeffs, power):
    if power == 1:
        return coeffs[:, :-1] * torch.tensor([7, 6, 5, 4, 3, 2, 1], dtype=torch.float32, device=torch.device("cuda"))
    elif power == 2:
        return coeffs[:, :-2] * torch.tensor([42, 30, 20, 12, 6, 2], dtype=torch.float32, device=torch.device("cuda"))
    elif power == 3:
        return coeffs[:, :-3] * torch.tensor([210, 120, 60, 24, 6], dtype=torch.float32, device=torch.device("cuda"))
    elif power == 4:
        return coeffs[:, :-4] * torch.tensor([840, 360, 120, 24], dtype=torch.float32, device=torch.device("cuda"))
    else:
        return coeffs


# Generate a septic segment
def get_coeffs(start_waypoint, end_waypoint, times, num_envs):
    coeffs_list = []

    start = {"pos": start_waypoint[:, :3], "vel": start_waypoint[:, 3:6], "acc": start_waypoint[:, 6:9]}
    end = {
        "pos": end_waypoint[:, :3],
        "vel": end_waypoint[:, 3:6],
        "acc": end_waypoint[:, 6:9],
        "jerk": end_waypoint[:, 9:12],
        "snap": end_waypoint[:, 12:15],
    }
    t_start = times[0]
    t_end = times[-1]

    coeffs_x, coeffs_y, coeffs_z = septic_trajectory(start, end, t_start, t_end, num_envs)

    coeffs_list.append((coeffs_x, coeffs_y, coeffs_z))

    return coeffs_list


# Compute trajectory at time t_eval
def evaluate_trajectory(coeffs_list, times, t_eval):
    for i in range(len(times) - 1):
        t_start = times[i]
        t_end = times[i + 1]

        if t_start <= t_eval <= t_end:
            T = t_eval - t_start
            coeffs_x, coeffs_y, coeffs_z = coeffs_list[i]
            # Compute position (5th order polynomial)
            position_x = torch.sum(
                coeffs_x
                * torch.tensor(
                    [T**7, T**6, T**5, T**4, T**3, T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")
                ),
                dim=-1,
                keepdim=True,
            )
            position_y = torch.sum(
                coeffs_y
                * torch.tensor(
                    [T**7, T**6, T**5, T**4, T**3, T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")
                ),
                dim=-1,
                keepdim=True,
            )
            position_z = torch.sum(
                coeffs_z
                * torch.tensor(
                    [T**7, T**6, T**5, T**4, T**3, T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")
                ),
                dim=-1,
                keepdim=True,
            )
            position = torch.cat([position_x, position_y, position_z], dim=-1)

            # Compute velocity
            velocity_x = torch.sum(
                compute_derivatives(coeffs_x, 1)
                * torch.tensor([T**6, T**5, T**4, T**3, T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")),
                dim=-1,
                keepdim=True,
            )
            velocity_y = torch.sum(
                compute_derivatives(coeffs_y, 1)
                * torch.tensor([T**6, T**5, T**4, T**3, T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")),
                dim=-1,
                keepdim=True,
            )
            velocity_z = torch.sum(
                compute_derivatives(coeffs_z, 1)
                * torch.tensor([T**6, T**5, T**4, T**3, T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")),
                dim=-1,
                keepdim=True,
            )
            velocity = torch.cat([velocity_x, velocity_y, velocity_z], dim=-1)

            # Compute acceleration
            acceleration_x = torch.sum(
                compute_derivatives(coeffs_x, 2)
                * torch.tensor([T**5, T**4, T**3, T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")),
                dim=-1,
                keepdim=True,
            )
            acceleration_y = torch.sum(
                compute_derivatives(coeffs_y, 2)
                * torch.tensor([T**5, T**4, T**3, T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")),
                dim=-1,
                keepdim=True,
            )
            acceleration_z = torch.sum(
                compute_derivatives(coeffs_z, 2)
                * torch.tensor([T**5, T**4, T**3, T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")),
                dim=-1,
                keepdim=True,
            )
            acceleration = torch.cat([acceleration_x, acceleration_y, acceleration_z], dim=-1)

            # Compute jerk
            jerk_x = torch.sum(
                compute_derivatives(coeffs_x, 3)
                * torch.tensor([T**4, T**3, T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")),
                dim=-1,
                keepdim=True,
            )
            jerk_y = torch.sum(
                compute_derivatives(coeffs_y, 3)
                * torch.tensor([T**4, T**3, T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")),
                dim=-1,
                keepdim=True,
            )
            jerk_z = torch.sum(
                compute_derivatives(coeffs_z, 3)
                * torch.tensor([T**4, T**3, T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")),
                dim=-1,
                keepdim=True,
            )
            jerk = torch.cat([jerk_x, jerk_y, jerk_z], dim=-1)

            # Compute snap
            snap_x = torch.sum(
                compute_derivatives(coeffs_x, 4)
                * torch.tensor([T**3, T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")),
                dim=-1,
                keepdim=True,
            )
            snap_y = torch.sum(
                compute_derivatives(coeffs_y, 4)
                * torch.tensor([T**3, T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")),
                dim=-1,
                keepdim=True,
            )
            snap_z = torch.sum(
                compute_derivatives(coeffs_z, 4)
                * torch.tensor([T**3, T**2, T, 1], dtype=torch.float32, device=torch.device("cuda")),
                dim=-1,
                keepdim=True,
            )
            snap = torch.cat([snap_x, snap_y, snap_z], dim=-1)

            return position, velocity, acceleration, jerk, snap

    raise ValueError("Time t_eval is out of range of the trajectory")
