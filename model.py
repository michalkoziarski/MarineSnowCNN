import tensorflow as tf


class MarineSnowCNN:
    def __init__(self, inputs, k=3, n_3d_layers=19, n_2d_layers=1, n_filters=64, n_input_channels=3,
                 n_output_channels=3, use_residual_connection=True, name='MarineSnowCNN'):
        if use_residual_connection:
            assert n_input_channels == n_output_channels

        self.inputs = inputs
        self.k = k
        self.n_3d_layers = n_3d_layers
        self.n_2d_layers = n_2d_layers
        self.n_layers = n_3d_layers + n_2d_layers
        self.n_filters = n_filters
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.use_residual_connection = use_residual_connection
        self.name = name
        self.weights = []
        self.biases = []
        self.outputs = self.inputs

        for i in range(self.n_layers):
            if i == 0:
                in_shape = self.n_input_channels
            else:
                in_shape = self.n_filters

            if i == self.n_layers - 1:
                out_shape = self.n_output_channels
            else:
                out_shape = self.n_filters

            if i < self.n_3d_layers:
                weight_shape = [self.k, self.k, self.k, in_shape, out_shape]
                convolution = tf.nn.conv3d
                strides = [1, 1, 1, 1, 1]
            else:
                weight_shape = [self.k, self.k, in_shape, out_shape]
                convolution = tf.nn.conv2d
                strides = [1, 1, 1, 1]

            weight = tf.get_variable('%s_weights_%d' % (self.name, i), weight_shape,
                                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('%s_biases_%d' % (self.name, i), [out_shape],
                                   initializer=tf.zeros_initializer())

            self.weights.append(weight)
            self.biases.append(bias)
            self.outputs = tf.nn.bias_add(convolution(self.outputs, weight, strides=strides, padding='SAME'), bias)

            if i == self.n_3d_layers - 1:
                self.outputs = tf.reduce_sum(self.outputs, axis=1)

            if i < self.n_layers - 1:
                self.outputs = tf.nn.relu(self.outputs)

        if self.use_residual_connection:
            self.outputs = tf.add(self.outputs, self.inputs[:, inputs.get_shape().as_list()[1] // 2])

        self.outputs = tf.minimum(tf.maximum(self.outputs, 0.0), 1.0)
