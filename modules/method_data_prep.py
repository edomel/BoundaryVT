import torch


def get_method_specific_data_dict(method, data_dict, device=torch.device("cuda:0")):

    out_dict = {}

    if method == "VT":
        out_dict["image"] = data_dict["image"].to(device)
        out_dict["mask"] = data_dict["field_mask"].to(device)
        out_dict["boundary"] = data_dict["field"].to(device)

    return out_dict
