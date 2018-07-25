import tensorflow as tf

class RNN():

    def __init__(self):
        self.num_layers = 2
        self.hidden_units = 128

    def create_model(self, inputs):
        with tf.variable_scope("rnn"):
            batch_size, seq_length, num_features = inputs.get_shape().as_list()

            def _get_cell():
                return tf.contrib.rnn.GRUCell(self.hidden_units)

            stacked_cells = tf.contrib.rnn.MultiRNNCell(
                [_get_cell() for _ in range(self.num_layers)], state_is_tuple=True
            )

            outputs, _ = tf.nn.dynamic_rnn(stacked_cells, inputs, dtype=tf.float32)

        if seq_length == None:
            seq_length = -1

        net = tf.reshape(outputs, (batch_size, seq_length, self.hidden_units))
        return net