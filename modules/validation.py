import torch

from medpy.metric.binary import assd
from modules.get_binary_boundary_mask import get_boundary_from_output


def validate(cfg, net, data_set, data_loader, device=torch.device("cuda:0")):

    with torch.no_grad():

        net.eval()
        edge_assd = 0.0

        for j, data_point in enumerate(data_loader):

            input_img = data_point["image"].to(device)
            batch, _, H, W = input_img.shape
            img_ids = data_point["img_id"]

            output_dict = net(input_img)
            target_boundary = data_point["boundary_mask"].to(device)

            for j in range(batch):

                boundary = target_boundary[j]
                predicted_boundary = output_dict["boundary_rep"][j]

                predicted_boundary = get_boundary_from_output(
                    cfg["METHOD_TYPE"], predicted_boundary, cfg["DIVERGENCE_THRESHOLD"]
                )

                predicted_boundary = (
                    torch.stack([predicted_boundary for k in range(3)], dim=-1) * 255
                )
                boundary = torch.stack([boundary for k in range(3)], dim=-1) * 255

                edge_assd += assd(
                    predicted_boundary.squeeze().cpu().numpy(),
                    boundary.squeeze().cpu().numpy(),
                )

        print("Validation mean assd", edge_assd / len(data_set))

    return edge_assd / len(data_set)
