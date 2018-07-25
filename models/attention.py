import tensorflow as tf

MAX_LENGTH = 1036

class Attention:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def pad_tensor(self, tensor):
        shape = tf.shape(tensor)
        padding = [[0, 0], [0, MAX_LENGTH - shape[1]], [0, 0]]
        result = tf.pad(tensor, padding, "CONSTANT")
        reshape = tf.reshape(result,(self.batch_size, MAX_LENGTH, 128))
        return reshape

    def loop_inputs(self,inputs, u):
        outputs_list = []
        for i in range(self.batch_size):
            outputs_list.append(self.attention_process(inputs[i], u))
        outputs = tf.stack(outputs_list)
        outputs = tf.reshape(outputs, (self.batch_size, 128))
        return outputs

    def attention_process(self, input, u):
        # input size is ? * 128
        u_y = tf.matmul(input, u, name='u_y')
        softmax = tf.nn.softmax(u_y, dim=0)
        output = tf.matmul(input, softmax, transpose_a=True)
        output_reshape = tf.reshape(output, (1, -1))
        return output_reshape

    def create_model(self, inputs):
        with tf.variable_scope("attention", reuse=tf.AUTO_REUSE):
            u = tf.get_variable('u', shape=(128, 1),initializer=tf.random_uniform_initializer(),dtype=tf.float32,
                                trainable=True)
            outputs = self.loop_inputs(inputs, u)
        return outputs