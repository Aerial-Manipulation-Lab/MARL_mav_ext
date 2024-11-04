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
