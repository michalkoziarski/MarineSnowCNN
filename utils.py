import model


def get_network(inputs, params):
    network = model.MarineSnowCNN(inputs, params['kernel_size'], params['n_3d_layers'], params['n_2d_layers'],
                                  params['n_filters'], n_output_channels=1, use_residual_connection=False)

    return network
