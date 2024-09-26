import torch
from torch.func import vmap

# @manual_batch
def off_diag(a: torch.Tensor) -> torch.Tensor:
    assert a.shape[0] == a.shape[1]
    n = a.shape[0]
    return (
        a.flatten(0, 1)[1:]
        .unflatten(0, (n - 1, n + 1))[:, :-1]
        .reshape(n, n - 1, *a.shape[2:])
    )

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


