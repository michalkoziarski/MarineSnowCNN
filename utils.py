import model
import numpy as np


def get_network(inputs, params):
    network = model.MarineSnowCNN(inputs, params['kernel_size'], params['n_3d_layers'], params['n_2d_layers'],
                                  params['n_filters'], n_output_channels=1, use_residual_connection=False)

    return network


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
