import torch
import torch.nn.functional as F


def zero_pixel_derivative_divergence(predicted_field):

    vector_field = F.normalize(predicted_field, p=2, dim=0)

    filter = torch.zeros((2, 1, 3, 3)).to(predicted_field.device)
    filter[0, 0] = torch.tensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]]).to(
        predicted_field.device
    )
    filter[1, 0] = torch.tensor([[0, 1, 0], [0, -1, 0], [0, 0, 0]]).to(
        predicted_field.device
    )
    divergence = torch.sum(
        F.conv2d(vector_field.unsqueeze(0), filter, padding=1, groups=2).squeeze(0),
        dim=0,
    )

    return divergence


def derivative_divergence(predicted_field):

    vector_field = F.normalize(predicted_field, p=2, dim=0)

    filter = torch.zeros((2, 1, 3, 3)).to(predicted_field.device)
    filter[0, 0] = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).to(
        predicted_field.device
    )
    filter[1, 0] = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).to(
        predicted_field.device
    )
    divergence = torch.sum(
        F.conv2d(vector_field.unsqueeze(0), filter, padding=1, groups=2).squeeze(0),
        dim=0,
    )

    return divergence


def integral_divergence(predicted_field):

    predicted_angle = torch.atan2(predicted_field[0], predicted_field[1])
    n_bins_mask = 8
    part_angle = (torch.acos(torch.zeros(1)).item()) / (n_bins_mask / 4.0)

    mask = (torch.arange(n_bins_mask) * part_angle).to(predicted_angle.device)
    mask = mask.unsqueeze(-1).repeat_interleave(predicted_angle.shape[-2], dim=-1)
    mask = mask.unsqueeze(-1).repeat_interleave(predicted_angle.shape[-1], dim=-1)

    projection = torch.cos(
        predicted_angle.unsqueeze(0).repeat_interleave(n_bins_mask, dim=0) - mask
    )

    filter = torch.zeros((1, n_bins_mask, 3, 3)).to(predicted_angle.device)
    filter[0, 0, 1, 2] = 1
    filter[0, 1, 0, 2] = 1
    filter[0, 2, 0, 1] = 1
    filter[0, 3, 0, 0] = 1
    filter[0, 4, 1, 0] = 1
    filter[0, 5, 2, 0] = 1
    filter[0, 6, 2, 1] = 1
    filter[0, 7, 2, 2] = 1

    divergence = F.conv2d(projection.unsqueeze(0), filter, padding=1).squeeze()

    return divergence
