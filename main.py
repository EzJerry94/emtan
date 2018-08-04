import tensorflow as tf
import utils
from tfrecord_generator import Generator
from training import Train
from data_provider import DataProvider
from models.cnn import CNN
from models.rnn import RNN
from models.fc import FC
from models.attention import Attention
from evaluation import Evaluation

class EMTAN():

    def __init__(self):
        # operation parameter
        self.operation = 'process_stats'
        # data source parameters
        self.arousal_train_tfrecords = './data/arousal_train_set.tfrecords'
        self.arousal_validate_tfrecords = './data/arousal_train_set.tfrecords'
        self.arousal_test_tfrecords = './data/arousal_train_set.tfrecords'
        self.valence_train_tfrecords = './data/arousal_train_set.tfrecords'
        self.valence_validate_tfrecords = './data/arousal_train_set.tfrecords'
        self.valence_test_tfrecords = './data/arousal_train_set.tfrecords'
        self.dominance_train_tfrecords = './data/arousal_train_set.tfrecords'
        self.dominance_validate_tfrecords = './data/arousal_train_set.tfrecords'
        self.dominance_test_tfrecords = './data/arousal_train_set.tfrecords'
        # train parameters
        self.is_multi = True
        self.is_arousal = False
        self.is_valence = False
        self.is_dominance = False
        # model parameters
        self.batch_size = 4
        self.epochs = 2
        self.num_classes = 3
        self.learning_rate = 1e-4
        self.is_attention = False

    def _reshape_to_conv(self, frames):
        frame_shape = frames.get_shape().as_list()
        num_featurs = frame_shape[-1]
        batch = -1
        frames = tf.reshape(frames, (batch, num_featurs))
        return frames

    def _reshape_to_rnn(self, frames):
        batch_size, num_features = frames.get_shape().as_list()
        seq_length = -1
        frames = tf.reshape(frames, [self.batch_size, seq_length, num_features])
        return frames

    def process_stats(self):
        original_file = './data/validation.csv'
        csv_file = './data/validation_set.csv'
        utils.preprocess_stats(original_file, csv_file)

    def tfrecords_generate(self):
        generator = Generator()
        generator.write_tfrecords()

    def get_data_provider(self):
        self.train_data_provider = DataProvider(self.train_tfrecords_folder, self.batch_size, True)
        self.validate_data_provider = DataProvider(self.validate_tfrecords_folder, self.batch_size, False)

    def training(self):
        predictions = self.get_predictions
        train = Train(self.train_data_provider, self.batch_size, self.epochs,
                      self.num_classes, self.learning_rate, predictions)
        train.start_training()

    def get_predictions(self, frames):
        frames = self._reshape_to_conv(frames)
        cnn = CNN()
        cnn_output = cnn.create_model(frames, cnn.conv_filters)
        cnn_output = self._reshape_to_rnn(cnn_output)
        rnn = RNN()
        rnn_output = rnn.create_model(cnn_output)
        if self.is_attention:
            attention = Attention(self.batch_size)
            attention_output = attention.create_model(rnn_output)
            fc = FC(self.num_classes)
            outputs = fc.create_model(attention_output)
        else:
            rnn_output = rnn_output[:, -1, :]
            fc = FC(self.num_classes)
            outputs = fc.create_model(rnn_output)
        return outputs

    def evaluation(self):
        predictions = self.get_predictions
        eval = Evaluation(self.validate_data_provider, self.batch_size, self.epochs,
                          self.num_classes, self.learning_rate, predictions)
        eval.start_evaluation()

def main():
    net = EMTAN()
    if net.operation == 'process_stats':
        #net.process_stats()
        utils.stats_distribution('./data/train_set.csv')
    elif net.operation == 'generate':
        net.tfrecords_generate()
    elif net.operation == 'training':
        net.get_data_provider()
        net.training()

if __name__ == '__main__':
    main()