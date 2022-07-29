import torch

from modules.compute_divergence import zero_pixel_derivative_divergence


def get_boundary_from_output(method_type, prediction, threshold=-1.0):

    if method_type == "VT":

        # divergence = derivative_divergence(angles)
        # divergence = integral_divergence(angles)
        divergence = zero_pixel_derivative_divergence(prediction)

        divergence[divergence > threshold] = 0
        divergence[divergence <= threshold] = 1

        # make prediction to mimic thickness of ground truth and for visualization
        divergence[1:] += divergence[:-1].clone()
        divergence[:, 1:] += divergence[:, :-1].clone()
        divergence[divergence > 1] = 1

        prediction = divergence.clone()

    return prediction
