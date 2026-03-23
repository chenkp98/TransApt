import torch
import numpy as np


def dihedral(p1, p2, p3, p4):
    eps = 1e-6
    a1 = p2 - p1
    a2 = p3 - p2
    a3 = p4 - p3

    v1 = torch.cross(a1, a2, dim=-1)
    v1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + eps)
    v2 = torch.cross(a2, a3, dim=-1)
    v2 = v2 / (torch.norm(v2, dim=-1, keepdim=True) + eps)

    sign = torch.sign(torch.sum(v1 * a3, dim=-1))
    sign[sign == 0.0] = 1.0

    cos_theta = torch.sum(v1 * v2, dim=-1) / (torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1) + eps)
    rad_vec = sign * torch.acos(torch.clip(cos_theta, -1.0, 1.0))
    return rad_vec


def compute_dihedrals(coords):
    X = torch.from_numpy(coords).float() if isinstance(coords, np.ndarray) else coords.float()

    n1 = dihedral(X[:, 0], X[:, 1], X[:, 2], X[:, 3]).unsqueeze(-1)
    n2 = dihedral(X[:, 2], X[:, 1], X[:, 0], X[:, 4]).unsqueeze(-1)
    n4 = dihedral(X[:, 0], X[:, 4], X[:, 5], X[:, 6]).unsqueeze(-1)

    D = torch.cat((n1, n2, n4), dim=-1)
    D_features = torch.cat((torch.cos(D), torch.sin(D)), dim=-1)
    return D_features.numpy()
