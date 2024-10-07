import functools
import torch
from torch.func import vmap


# @manual_batch
def off_diag(a: torch.Tensor) -> torch.Tensor:
    assert a.shape[0] == a.shape[1]
    n = a.shape[0]
    return a.flatten(0, 1)[1:].unflatten(0, (n - 1, n + 1))[:, :-1].reshape(n, n - 1, *a.shape[2:])


# @manual_batch
def cpos(p1: torch.Tensor, p2: torch.Tensor):
    assert p1.shape[1] == p2.shape[1]
    return p1.unsqueeze(1) - p2.unsqueeze(0)


def get_drone_rpos(drone_pos):
    drone_rpos = vmap(cpos)(drone_pos, drone_pos)
    drone_rpos = vmap(off_diag)(drone_rpos)
    return drone_rpos


def get_drone_pdist(drone_rpos):
    return torch.norm(drone_rpos, dim=-1)


def manual_batch(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        batch_shapes = {arg.shape[:-1] for arg in args if isinstance(arg, torch.Tensor)}
        if not len(batch_shapes) == 1:
            raise ValueError
        batch_shape = batch_shapes.pop()
        args = (arg.reshape(-1, arg.shape[-1]) if isinstance(arg, torch.Tensor) else arg for arg in args)
        kwargs = {k: v.reshape(-1, v.shape[-1]) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        out = func(*args, **kwargs)
        return out.unflatten(0, batch_shape)

    return wrapped


@manual_batch
def quat_rotate(q: torch.Tensor, v: torch.Tensor):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@manual_batch
def quat_axis(q: torch.Tensor, axis: int = 0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)

import torch

# Quintic polynomial trajectory for a single segment (3D)
def quintic_trajectory_3d(start, end, t_start, t_end):
    T = t_end - t_start
    A = torch.tensor([
        [0, 0, 0, 0, 0, 1],             # p(t0)
        [T**5, T**4, T**3, T**2, T, 1],  # p(t1)
        [0, 0, 0, 0, 1, 0],             # v(t0)
        [5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],  # v(t1)
        [0, 0, 0, 2, 0, 0],             # a(t0)
        [20*T**3, 12*T**2, 6*T, 2, 0, 0]     # a(t1)
    ], dtype=torch.float32)
    
    # Each axis x, y, z has its own polynomial
    b_x = torch.tensor([start['pos'][0], end['pos'][0], start['vel'][0], end['vel'][0], start['acc'][0], end['acc'][0]], dtype=torch.float32)
    b_y = torch.tensor([start['pos'][1], end['pos'][1], start['vel'][1], end['vel'][1], start['acc'][1], end['acc'][1]], dtype=torch.float32)
    b_z = torch.tensor([start['pos'][2], end['pos'][2], start['vel'][2], end['vel'][2], start['acc'][2], end['acc'][2]], dtype=torch.float32)
    
    # Solve for polynomial coefficients for each axis
    coeffs_x = torch.linalg.solve(A, b_x)
    coeffs_y = torch.linalg.solve(A, b_y)
    coeffs_z = torch.linalg.solve(A, b_z)
    
    return coeffs_x, coeffs_y, coeffs_z

# Generate a minimum snap trajectory for multiple 3D waypoints
def minimum_snap_spline_3d(waypoints, times):
    n_points = len(waypoints)
    coeffs_list = []
    
    # Loop through each pair of waypoints to create a segment
    for i in range(n_points - 1):
        start = {'pos': waypoints[i], 'vel': torch.zeros(3), 'acc': torch.zeros(3)}  # Assuming zero velocity/acceleration
        end = {'pos': waypoints[i+1], 'vel': torch.zeros(3), 'acc': torch.zeros(3)}
        t_start = times[i]
        t_end = times[i+1]
        
        coeffs_x, coeffs_y, coeffs_z = quintic_trajectory_3d(start, end, t_start, t_end)
        coeffs_list.append((coeffs_x, coeffs_y, coeffs_z))
    
    return coeffs_list

# Compute derivatives of the polynomial (velocity, acceleration, jerk, snap)
def compute_derivatives_3d(coeffs, power):
    if power == 1:
        return coeffs[:-1] * torch.tensor([5, 4, 3, 2, 1], dtype=torch.float32)
    elif power == 2:
        return coeffs[:-2] * torch.tensor([20, 12, 6, 2], dtype=torch.float32)
    elif power == 3:
        return coeffs[:-3] * torch.tensor([60, 24, 6], dtype=torch.float32)
    elif power == 4:
        return coeffs[:-4] * torch.tensor([120, 24], dtype=torch.float32)
    else:
        return coeffs

# Evaluate the trajectory and its derivatives (position, velocity, acceleration, jerk, snap) at a specific time
def evaluate_trajectory_3d(coeffs_list, times, t_eval):
    for i in range(len(times) - 1):
        t_start = times[i]
        t_end = times[i+1]
        
        if t_start <= t_eval <= t_end:
            T = t_eval - t_start
            coeffs_x, coeffs_y, coeffs_z = coeffs_list[i]
            
            # Compute position
            position_x = torch.sum(coeffs_x * torch.tensor([T**5, T**4, T**3, T**2, T, 1], dtype=torch.float32))
            position_y = torch.sum(coeffs_y * torch.tensor([T**5, T**4, T**3, T**2, T, 1], dtype=torch.float32))
            position_z = torch.sum(coeffs_z * torch.tensor([T**5, T**4, T**3, T**2, T, 1], dtype=torch.float32))
            position = torch.tensor([position_x, position_y, position_z], dtype=torch.float32)
            
            # Compute velocity
            velocity_x = torch.sum(compute_derivatives_3d(coeffs_x, 1) * torch.tensor([T**4, T**3, T**2, T, 1], dtype=torch.float32))
            velocity_y = torch.sum(compute_derivatives_3d(coeffs_y, 1) * torch.tensor([T**4, T**3, T**2, T, 1], dtype=torch.float32))
            velocity_z = torch.sum(compute_derivatives_3d(coeffs_z, 1) * torch.tensor([T**4, T**3, T**2, T, 1], dtype=torch.float32))
            velocity = torch.tensor([velocity_x, velocity_y, velocity_z], dtype=torch.float32)
            
            # Compute acceleration
            acceleration_x = torch.sum(compute_derivatives_3d(coeffs_x, 2) * torch.tensor([T**3, T**2, T, 1], dtype=torch.float32))
            acceleration_y = torch.sum(compute_derivatives_3d(coeffs_y, 2) * torch.tensor([T**3, T**2, T, 1], dtype=torch.float32))
            acceleration_z = torch.sum(compute_derivatives_3d(coeffs_z, 2) * torch.tensor([T**3, T**2, T, 1], dtype=torch.float32))
            acceleration = torch.tensor([acceleration_x, acceleration_y, acceleration_z], dtype=torch.float32)
            
            # Compute jerk
            jerk_x = torch.sum(compute_derivatives_3d(coeffs_x, 3) * torch.tensor([T**2, T, 1], dtype=torch.float32))
            jerk_y = torch.sum(compute_derivatives_3d(coeffs_y, 3) * torch.tensor([T**2, T, 1], dtype=torch.float32))
            jerk_z = torch.sum(compute_derivatives_3d(coeffs_z, 3) * torch.tensor([T**2, T, 1], dtype=torch.float32))
            jerk = torch.tensor([jerk_x, jerk_y, jerk_z], dtype=torch.float32)
            
            # Compute snap
            snap_x = torch.sum(compute_derivatives_3d(coeffs_x, 4) * torch.tensor([T, 1], dtype=torch.float32))
            snap_y = torch.sum(compute_derivatives_3d(coeffs_y, 4) * torch.tensor([T, 1], dtype=torch.float32))
            snap_z = torch.sum(compute_derivatives_3d(coeffs_z, 4) * torch.tensor([T, 1], dtype=torch.float32))
            snap = torch.tensor([snap_x, snap_y, snap_z], dtype=torch.float32)
            
            return position, velocity, acceleration, jerk, snap
    
    return None
