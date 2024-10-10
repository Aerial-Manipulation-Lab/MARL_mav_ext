import torch
from omni.isaac.lab.utils.math import quat_inv, quat_mul

# Quintic polynomial trajectory for a single segment (3D)
def quintic_trajectory(start, end, t_start, t_end):
    T = t_end - t_start
    A = torch.tensor([
        [0, 0, 0, 0, 0, 1],             # p(t0)
        [T**5, T**4, T**3, T**2, T, 1],  # p(t1)
        [0, 0, 0, 0, 1, 0],             # v(t0)
        [5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],  # v(t1)
        [0, 0, 0, 2, 0, 0],             # a(t0)
        [20*T**3, 12*T**2, 6*T, 2, 0, 0]     # a(t1)
    ], dtype=torch.float32)
    
    # Position polynomials
    b_x = torch.tensor([start['pos'][0], end['pos'][0], start['vel'][0], end['vel'][0], start['acc'][0], end['acc'][0]], dtype=torch.float32)
    b_y = torch.tensor([start['pos'][1], end['pos'][1], start['vel'][1], end['vel'][1], start['acc'][1], end['acc'][1]], dtype=torch.float32)
    b_z = torch.tensor([start['pos'][2], end['pos'][2], start['vel'][2], end['vel'][2], start['acc'][2], end['acc'][2]], dtype=torch.float32)
    
    coeffs_x = torch.linalg.solve(A, b_x)
    coeffs_y = torch.linalg.solve(A, b_y)
    coeffs_z = torch.linalg.solve(A, b_z)

    return coeffs_x, coeffs_y, coeffs_z

# Compute derivatives of the polynomial (velocity, acceleration, jerk, snap)
def compute_derivatives(coeffs, power):
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

# Generate a minimum snap trajectory for multiple 3D waypoints with orientation
def minimum_snap_spline(waypoints, times):

    n_points = len(times)
    waypoint_id_increment = len(waypoints) // n_points
    coeffs_list = []
    orientations = []

    for i in range(n_points - 1):
        curr_point_id = i * waypoint_id_increment
        start = {'pos': waypoints[curr_point_id:curr_point_id+3], 'vel': waypoints[curr_point_id + 3: curr_point_id + 6],
                  'acc': waypoints[curr_point_id + 6: curr_point_id + 9], 'orientation': waypoints[curr_point_id + 9: curr_point_id + 13]}
        next_point_id = (i + 1) * waypoint_id_increment
        end = {'pos': waypoints[next_point_id:next_point_id+3], 'vel': waypoints[next_point_id + 3: next_point_id + 6],
                'acc': waypoints[next_point_id + 6: next_point_id + 9], 'orientation': waypoints[next_point_id + 9: next_point_id + 13]}
        t_start = times[i]
        t_end = times[i+1]
        
        # Calculate the position trajectory (quintic polynomial)
        coeffs_x, coeffs_y, coeffs_z = quintic_trajectory(start, end, t_start, t_end)
        coeffs_list.append((coeffs_x, coeffs_y, coeffs_z))

        # SLERP for orientation over this segment
        orientation_traj = (start['orientation'], end['orientation'])
        orientations.append(orientation_traj)
    
    return coeffs_list, orientations

# Compute trajectory, SLERP orientation, and angular velocity at time t_eval
def evaluate_trajectory(coeffs_list, orientations, times, t_eval):
    for i in range(len(times) - 1):
        t_start = times[i]
        t_end = times[i+1]
        
        if t_start <= t_eval <= t_end:
            T = t_eval - t_start
            coeffs_x, coeffs_y, coeffs_z = coeffs_list[i]
            
            # Compute position (5th order polynomial)
            position_x = torch.sum(coeffs_x * torch.tensor([T**5, T**4, T**3, T**2, T, 1], dtype=torch.float32))
            position_y = torch.sum(coeffs_y * torch.tensor([T**5, T**4, T**3, T**2, T, 1], dtype=torch.float32))
            position_z = torch.sum(coeffs_z * torch.tensor([T**5, T**4, T**3, T**2, T, 1], dtype=torch.float32))
            position = torch.tensor([position_x, position_y, position_z], dtype=torch.float32, device=torch.device('cuda'))
            
            # Compute velocity
            velocity_x = torch.sum(compute_derivatives(coeffs_x, 1) * torch.tensor([T**4, T**3, T**2, T, 1], dtype=torch.float32))
            velocity_y = torch.sum(compute_derivatives(coeffs_y, 1) * torch.tensor([T**4, T**3, T**2, T, 1], dtype=torch.float32))
            velocity_z = torch.sum(compute_derivatives(coeffs_z, 1) * torch.tensor([T**4, T**3, T**2, T, 1], dtype=torch.float32))
            velocity = torch.tensor([velocity_x, velocity_y, velocity_z], dtype=torch.float32, device=torch.device('cuda'))
            
            # Compute acceleration
            acceleration_x = torch.sum(compute_derivatives(coeffs_x, 2) * torch.tensor([T**3, T**2, T, 1], dtype=torch.float32))
            acceleration_y = torch.sum(compute_derivatives(coeffs_y, 2) * torch.tensor([T**3, T**2, T, 1], dtype=torch.float32))
            acceleration_z = torch.sum(compute_derivatives(coeffs_z, 2) * torch.tensor([T**3, T**2, T, 1], dtype=torch.float32))
            acceleration = torch.tensor([acceleration_x, acceleration_y, acceleration_z], dtype=torch.float32, device=torch.device('cuda'))
            
            # Compute jerk
            jerk_x = torch.sum(compute_derivatives(coeffs_x, 3) * torch.tensor([T**2, T, 1], dtype=torch.float32))
            jerk_y = torch.sum(compute_derivatives(coeffs_y, 3) * torch.tensor([T**2, T, 1], dtype=torch.float32))
            jerk_z = torch.sum(compute_derivatives(coeffs_z, 3) * torch.tensor([T**2, T, 1], dtype=torch.float32))
            jerk = torch.tensor([jerk_x, jerk_y, jerk_z], dtype=torch.float32, device=torch.device('cuda'))
            
            # Compute snap
            snap_x = torch.sum(compute_derivatives(coeffs_x, 4) * torch.tensor([T, 1], dtype=torch.float32))
            snap_y = torch.sum(compute_derivatives(coeffs_y, 4) * torch.tensor([T, 1], dtype=torch.float32))
            snap_z = torch.sum(compute_derivatives(coeffs_z, 4) * torch.tensor([T, 1], dtype=torch.float32))
            snap = torch.tensor([snap_x, snap_y, snap_z], dtype=torch.float32, device=torch.device('cuda'))
            
            # Angular velocity (finite difference approximation)
            delta_t = t_end - t_start
            
            return position, velocity, acceleration, jerk, snap
    return None