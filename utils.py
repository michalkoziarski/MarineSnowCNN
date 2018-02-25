import model


def get_network(inputs, params):
    if params['annotation_type'] == 'mask':
        n_output_channels = 1
        use_residual_connection = False
    elif params['annotation_type'] == 'filtered':
        n_output_channels = 3
        use_residual_connection = True
    else:
        raise NotImplementedError

    network = model.MarineSnowCNN(inputs, params['kernel_size'], params['n_3d_layers'], params['n_2d_layers'],
                                  params['n_filters'], n_output_channels=n_output_channels,
                                  use_residual_connection=use_residual_connection)

    return network
