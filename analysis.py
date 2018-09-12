import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools

class Analysis:

    def __init__(self, eval_data_provider, batch_size, epochs, num_classes,
                 learning_rate, predictions):
        self.eval_data_provider = eval_data_provider
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.eval_sample_num = 1811
        self.predictions = predictions
        self.model_path = './ckpt/multi/multi_model.ckpt'

    def start_evaluation(self):
        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(3)  # set random seed for initialization

            self.eval_data_provider.get_batch()

            iterator = self.eval_data_provider.dataset.make_initializable_iterator()

            frames, arousals, valences, dominances, files = iterator.get_next()

            frames = tf.reshape(frames, (self.batch_size, -1, 640))

            arousals_attention, valences_attention, dominances_attention = self.predictions(frames)

            saver = tf.train.Saver()

        with tf.Session(graph=g) as sess:
            eval_num_batches = int(self.eval_sample_num / self.batch_size)
            saver.restore(sess, self.model_path)
            print("Model restored.")

            print('\n Start Validation for epoch \n')
            sess.run(iterator.initializer)

            for batch in range(eval_num_batches - 1):
                print('Example {}/{}'.format(batch + 1, eval_num_batches))
                files_val = sess.run([files])
                print(files_val)