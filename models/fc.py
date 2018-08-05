import tensorflow as tf

slim = tf.contrib.slim

class FC():

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def create_model(self, inputs, scope):
        with tf.variable_scope(scope):
            fc = slim.layers.linear(inputs, self.num_classes)
        return fc