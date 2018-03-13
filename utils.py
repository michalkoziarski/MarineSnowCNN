import model
import numpy as np


def get_network(inputs, params):
    network = model.MarineSnowCNN(inputs, params['kernel_size'], params['n_3d_layers'], params['n_2d_layers'],
                                  params['n_filters'], n_output_channels=1, use_residual_connection=False)

    return network


def peripheral_median_filter(input, mask, kernel_size, threshold=0.5):
    assert len(input.shape) == 4
    assert input.shape[0] >= 3

    output = np.empty(input.shape[1:], dtype=input.dtype)

    for i in range(input.shape[1]):
        for j in range(input.shape[2]):
            for channel in range(input.shape[3]):
                if mask[i, j] >= threshold:
                    x_start = np.max((i - kernel_size // 2, 0))
                    x_end = i + kernel_size // 2 + 1
                    y_start = np.max((j - kernel_size // 2, 0))
                    y_end = j + kernel_size // 2 + 1
                    temporal_range = [input.shape[0] // 2 - 1, input.shape[0] // 2 + 1]

                    output[i, j, channel] = np.median(input[temporal_range, x_start:x_end, y_start:y_end, channel])
                else:
                    output[i, j, channel] = input[input.shape[0] // 2, i, j, channel]

    return output


def adaptive_median_filter(input, mask, initial_kernel_size=1, threshold=0.5):
    assert len(input.shape) == 4

    central_frame = input[input.shape[0] // 2]
    output = np.empty(input.shape[1:], dtype=input.dtype)

    for i in range(input.shape[1]):
        for j in range(input.shape[2]):
            k = initial_kernel_size

            while True:
                x_start = np.max((i - k // 2, 0))
                x_end = i + k // 2 + 1
                y_start = np.max((j - k // 2, 0))
                y_end = j + k // 2 + 1

                submask = (mask[x_start:x_end, y_start:y_end] < threshold)[:, :, 0]

                if np.sum(submask) > 0:
                    for channel in range(input.shape[3]):
                        current_slice = central_frame[x_start:x_end, y_start:y_end, channel]
                        output[i, j, channel] = np.median(current_slice[submask])

                    break
                else:
                    k += 2

    return output
