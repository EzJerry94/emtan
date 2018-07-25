import tensorflow as tf

slim = tf.contrib.slim

class CNN():

    def __init__(self):
        self.is_training = True
        self.conv_filters = 40

    def create_model(self, frames, conv_filters):
        with tf.variable_scope("cnn"):
            batch_size, num_features = frames.get_shape().as_list()
            shape = ([-1, 1, num_features, 1])
            audio_input = tf.reshape(frames, shape)

            with slim.arg_scope([slim.layers.conv2d], padding='SAME'):
                #net = slim.dropout(audio_input, is_training=self.is_training)
                net = slim.layers.conv2d(audio_input, conv_filters, (1, 20))

                # Subsampling of the signal to 8Khz
                net = tf.nn.max_pool(
                    net,
                    ksize=[1, 1, 2, 1],
                    strides=[1, 1, 2, 1],
                    padding='SAME',
                    name='pool1'
                )

                # Original model had 400 output filters for the second conv layer
                # but this trains much faster and achieves comparable accuracy.
                net = slim.layers.conv2d(net, conv_filters, (1, 40))
                net = tf.reshape(net, (-1, num_features // 2, conv_filters, 1))

                # Pooling over the feature maps.
                net = tf.nn.max_pool(
                    net,
                    ksize=[1, 1, 10, 1],
                    strides=[1, 1, 10, 1],
                    padding='SAME',
                    name='pool2'
                )

            net = tf.reshape(net, (-1, num_features // 2 * 4))

        return net