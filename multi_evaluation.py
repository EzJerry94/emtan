import tensorflow as tf
from sklearn.metrics import recall_score
import numpy as np

class Evaluation:

    def __init__(self, eval_data_provider, batch_size, epochs, num_classes,
                 learning_rate, predictions):
        self.eval_data_provider = eval_data_provider
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.eval_sample_num = 1811
        self.predictions = predictions
        self.model_path = './ckpt/single/arousal/model.ckpt'

    def start_evaluation(self):
        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(3)  # set random seed for initialization

            self.eval_data_provider.get_batch()

            iterator = self.eval_data_provider.dataset.make_initializable_iterator()
            frames, labels = iterator.get_next()

            labels = tf.one_hot(labels, depth=3, axis=-1)
            labels = tf.reshape(labels, (self.batch_size, self.num_classes))
            frames = tf.reshape(frames, (self.batch_size, -1, 640))

            eval_prediction = self.predictions(frames)
            saver = tf.train.Saver()

        with tf.Session(graph=g) as sess:
            eval_num_batches = int(self.eval_sample_num / self.batch_size)
            saver.restore(sess, self.model_path)
            print("Model restored.")

            print('\n Start Validation for epoch \n')
            sess.run(iterator.initializer)
            eval_predictions_list = []
            eval_labels_list = []
            for batch in range(eval_num_batches):
                print('Example {}/{}'.format(batch + 1, eval_num_batches))
                preds, labs, = sess.run([eval_prediction, labels])
                eval_predictions_list.append(preds)
                eval_labels_list.append(labs)

            eval_predictions_list = np.reshape(eval_predictions_list, (-1, self.num_classes))
            eval_labels_list = np.reshape(eval_labels_list, (-1, self.num_classes))
            eval_predictions_list = np.argmax(eval_predictions_list, axis=1)
            eval_labels_list = np.argmax(eval_labels_list, axis=1)

            mean_eval = recall_score(eval_labels_list, eval_predictions_list, average="macro")
            print("uar: ", mean_eval)