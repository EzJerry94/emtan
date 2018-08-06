import tensorflow as tf
from sklearn.metrics import recall_score
import numpy as np

class MultiEvaluation:

    def __init__(self, eval_data_provider, batch_size, num_classes,predictions):
        self.eval_data_provider = eval_data_provider
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.eval_sample_num = 1811
        self.predictions = predictions
        self.model_path = './ckpt/multi/multi_model.ckpt'

    def start_evaluation(self):
        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(3)  # set random seed for initialization

            self.eval_data_provider.get_batch()

            iterator = self.eval_data_provider.dataset.make_initializable_iterator()

            frames, arousals, valences, dominances = iterator.get_next()

            arousals = tf.one_hot(arousals, depth=3, axis=-1)
            arousals = tf.reshape(arousals, (self.batch_size, self.num_classes))
            valences = tf.one_hot(valences, depth=3, axis=-1)
            valences = tf.reshape(valences, (self.batch_size, self.num_classes))
            dominances = tf.one_hot(dominances, depth=3, axis=-1)
            dominances = tf.reshape(dominances, (self.batch_size, self.num_classes))

            frames = tf.reshape(frames, (self.batch_size, -1, 640))

            arousals_prediction, valences_prediction, dominances_prediction = self.predictions(frames)

            saver = tf.train.Saver()

        with tf.Session(graph=g) as sess:
            eval_num_batches = int(self.eval_sample_num / self.batch_size)
            saver.restore(sess, self.model_path)
            print("Model restored.")

            print('\n Start Validation for epoch \n')
            sess.run(iterator.initializer)
            eval_arousal_predictions_list = []
            eval_arousal_labels_list = []
            eval_valence_predictions_list = []
            eval_valence_labels_list = []
            eval_dominance_predictions_list = []
            eval_dominance_labels_list = []
            for batch in range(eval_num_batches):
                print('Example {}/{}'.format(batch + 1, eval_num_batches))
                arousal_preds, arousal_labs, valence_preds, valence_labs,\
                    dominance_preds, dominance_labs= sess.run([arousals_prediction, arousals,
                                                               valences_prediction, valences,
                                                               dominances_prediction, dominances])
                eval_arousal_predictions_list.append(arousal_preds)
                eval_arousal_labels_list.append(arousal_labs)
                eval_valence_predictions_list.append(valence_preds)
                eval_valence_labels_list.append(valence_labs)
                eval_dominance_predictions_list.append(dominance_preds)
                eval_dominance_labels_list.append(dominance_labs)

            eval_arousal_predictions_list = np.reshape(eval_arousal_predictions_list, (-1, self.num_classes))
            eval_arousal_labels_list = np.reshape(eval_arousal_labels_list, (-1, self.num_classes))
            eval_arousal_predictions_list = np.argmax(eval_arousal_predictions_list, axis=1)
            eval_arousal_labels_list = np.argmax(eval_arousal_labels_list, axis=1)

            eval_valence_predictions_list = np.reshape(eval_valence_predictions_list, (-1, self.num_classes))
            eval_valence_labels_list = np.reshape(eval_valence_labels_list, (-1, self.num_classes))
            eval_valence_predictions_list = np.argmax(eval_valence_predictions_list, axis=1)
            eval_valence_labels_list = np.argmax(eval_valence_labels_list, axis=1)

            eval_dominance_predictions_list = np.reshape(eval_dominance_predictions_list, (-1, self.num_classes))
            eval_dominance_labels_list = np.reshape(eval_dominance_labels_list, (-1, self.num_classes))
            eval_dominance_predictions_list = np.argmax(eval_dominance_predictions_list, axis=1)
            eval_dominance_labels_list = np.argmax(eval_dominance_labels_list, axis=1)

            arousal_mean_eval = recall_score(eval_arousal_labels_list, eval_arousal_predictions_list, average="macro")
            valence_mean_eval = recall_score(eval_valence_labels_list, eval_valence_predictions_list, average="macro")
            dominance_mean_eval = recall_score(eval_dominance_labels_list, eval_dominance_predictions_list, average="macro")
            print("arousal uar: ", arousal_mean_eval)
            print("valence uar: ", valence_mean_eval)
            print("dominance uar: ", dominance_mean_eval)