import tensorflow as tf
import utils
from tfrecord_generator import Generator
from tfrecord_generator_old import SingleGenerator
from multi_training import MultiTrain
from multi_data_provider import MultiDataProvider
from data_provider import DataProvider
from training import Train
from multi_evaluation import MultiEvaluation
from evaluation import Evaluation
from models.cnn import CNN
from models.rnn import RNN
from models.fc import FC
from models.attention import Attention
from evaluation import Evaluation

class EMTAN():

    def __init__(self):
        # operation parameter
        self.operation = 'training'
        # data source parameters
        self.arousal_train_tfrecords = './data/arousal/train_set.tfrecords'
        self.arousal_validate_tfrecords = './data/arousal/validate_set.tfrecords'
        self.arousal_test_tfrecords = './data/arousal/test_set.tfrecords'
        self.valence_train_tfrecords = './data/valence/train_set.tfrecords'
        self.valence_validate_tfrecords = './data/valence/validate_set.tfrecords'
        self.valence_test_tfrecords = './data/valence/test_set.tfrecords'
        self.dominance_train_tfrecords = './data/dominance/train_set.tfrecords'
        self.dominance_validate_tfrecords = './data/dominance/validate_set.tfrecords'
        self.dominance_test_tfrecords = './data/dominance/test_set.tfrecords'
        self.multi_train_tfrecords = './data/multi/multi_train_set.tfrecords'
        self.multi_validation_tfrecords = './data/multi/multi_validation_set.tfrecords'
        self.multi_test_tfrecords = './data/multi/multi_test_set.tfrecords'
        # train parameters
        self.is_multi = True
        self.is_arousal = False
        self.is_valence = False
        self.is_dominance = False
        # model parameters
        self.batch_size = 4
        self.epochs = 50
        self.num_classes = 3
        self.learning_rate = 5e-5
        self.is_attention = True
        self.keep_prob = 0.5

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
        original_file = './data/raw/test.csv'
        csv_file = './data/raw/test_set.csv'
        utils.preprocess_stats(original_file, csv_file)

    def tfrecords_generate(self):
        generator = Generator()
        generator.write_multi_tfrecords()

    def arousal_tfrecords_generate(self):
        train_generator = SingleGenerator('arousal', 'data/raw/train_set.csv', self.arousal_train_tfrecords, True)
        train_generator.write_tfrecords()
        validate_generator = SingleGenerator('arousal', 'data/raw/validation_set.csv', self.arousal_validate_tfrecords, False)
        validate_generator.write_tfrecords()
        test_generator = SingleGenerator('arousal', 'data/raw/test_set.csv', self.arousal_test_tfrecords, False)
        test_generator.write_tfrecords()

    def valence_tfrecords_generate(self):
        train_generator = SingleGenerator('valence', 'data/raw/train_set.csv', self.valence_train_tfrecords, True)
        train_generator.write_tfrecords()
        validate_generator = SingleGenerator('valence', 'data/raw/validation_set.csv', self.valence_validate_tfrecords, False)
        validate_generator.write_tfrecords()
        test_generator = SingleGenerator('valence', 'data/raw/test_set.csv', self.valence_test_tfrecords, False)
        test_generator.write_tfrecords()

    def dominance_tfrecords_generate(self):
        train_generator = SingleGenerator('dominance', 'data/raw/train_set.csv', self.dominance_train_tfrecords, True)
        train_generator.write_tfrecords()
        validate_generator = SingleGenerator('dominance', 'data/raw/validation_set.csv', self.dominance_validate_tfrecords, False)
        validate_generator.write_tfrecords()
        test_generator = SingleGenerator('dominance', 'data/raw/test_set.csv', self.dominance_test_tfrecords, False)
        test_generator.write_tfrecords()

    def get_multi_train_data_provider(self):
        self.multi_train_data_provider = MultiDataProvider(self.multi_train_tfrecords, self.batch_size, True)

    def get_multi_validation_data_provider(self):
        self.multi_validation_data_provider = MultiDataProvider(self.multi_validation_tfrecords, self.batch_size, False)

    def get_multi_test_data_provider(self):
        self.multi_test_data_provider = MultiDataProvider(self.multi_test_tfrecords, self.batch_size, False)

    def get_multi_predictions(self, frames):
        frames = self._reshape_to_conv(frames)
        cnn = CNN()
        if self.operation == 'training':
            cnn_output = cnn.create_model(frames, cnn.conv_filters, keep_prob=self.keep_prob)
        else:
            cnn_output = cnn.create_model(frames, cnn.conv_filters, keep_prob=1.0)
        cnn_output = self._reshape_to_rnn(cnn_output)
        rnn = RNN()
        arousal_rnn_output = rnn.create_model(cnn_output, 'arousal_rnn')
        valence_rnn_output = rnn.create_model(cnn_output, 'valence_rnn')
        dominance_rnn_output = rnn.create_model(cnn_output, 'dominance_rnn')
        if self.is_attention:
            attention = Attention(self.batch_size)
            arousal_attention_output = attention.create_model(arousal_rnn_output, 'arousal_attention')
            valence_attention_output = attention.create_model(valence_rnn_output, 'valence_attention')
            dominance_attention_output = attention.create_model(dominance_rnn_output, 'dominance_attention')
            fc = FC(self.num_classes)
            arousal_fc_outputs = fc.create_model(arousal_attention_output, 'arousal_fc')
            valence_fc_outputs = fc.create_model(valence_attention_output, 'valence_fc')
            dominance_fc_outputs = fc.create_model(dominance_attention_output, 'dominance_fc')
        else:
            arousal_rnn_output = arousal_rnn_output[:, -1, :]
            valence_rnn_output = valence_rnn_output[:, -1, :]
            dominance_rnn_output = dominance_rnn_output[:, -1, :]
            fc = FC(self.num_classes)
            arousal_fc_outputs = fc.create_model(arousal_rnn_output, 'arousal_fc')
            valence_fc_outputs = fc.create_model(valence_rnn_output, 'valence_fc')
            dominance_fc_outputs = fc.create_model(dominance_rnn_output, 'dominance_fc')

        return arousal_fc_outputs, valence_fc_outputs, dominance_fc_outputs

    def multi_task_training(self):
        self.get_multi_train_data_provider()
        predictions = self.get_multi_predictions
        train = MultiTrain(self.multi_train_data_provider, self.batch_size, self.epochs,
                      self.num_classes, self.learning_rate, predictions)
        train.start_training()

    def multi_task_validation(self):
        self.get_multi_validation_data_provider()
        predictions = self.get_multi_predictions
        validation = MultiEvaluation(self.multi_validation_data_provider, self.batch_size, self.num_classes, predictions)
        validation.start_evaluation()

    def get_single_train_data_provider(self, train_tfrecords):
        self.single_train_data_provider = DataProvider(train_tfrecords, self.batch_size, True)

    def get_single_validate_data_provider(self, validate_tfrecords):
        self.single_validate_data_provider = DataProvider(validate_tfrecords, self.batch_size, False)

    def get_single_test_data_provider(self, test_tfrecords):
        self.single_test_data_provider = DataProvider(test_tfrecords, self.batch_size, False)

    def arousal_training(self):
        self.get_single_train_data_provider(self.arousal_train_tfrecords)
        predictions = self.get_predictions
        train = Train(self.single_train_data_provider, self.batch_size, self.epochs, self.num_classes,
                      self.learning_rate, predictions, 9051, './ckpt/arousal/model.ckpt', 'arousal')
        train.start_training()

    def arousal_validation(self):
        self.get_single_validate_data_provider(self.arousal_validate_tfrecords)
        predictions = self.get_predictions
        validation = Evaluation(self.single_validate_data_provider, self.batch_size, self.epochs, self.num_classes,
                                self.learning_rate, predictions, 1811, 'arousal')
        validation.start_evaluation()


    def get_predictions(self, frames, scope):
        frames = self._reshape_to_conv(frames)
        cnn = CNN()
        if self.operation == 'training':
            cnn_output = cnn.create_model(frames, cnn.conv_filters, keep_prob=self.keep_prob)
        else:
            cnn_output = cnn.create_model(frames, cnn.conv_filters, keep_prob=1.0)
        cnn_output = self._reshape_to_rnn(cnn_output)
        rnn = RNN()
        rnn_output = rnn.create_model(cnn_output, scope + '_rnn')
        if self.is_attention:
            attention = Attention(self.batch_size)
            attention_output = attention.create_model(rnn_output, scope + '_attention')
            fc = FC(self.num_classes)
            outputs = fc.create_model(attention_output, scope + '_fc')
        else:
            rnn_output = rnn_output[:, -1, :]
            fc = FC(self.num_classes)
            outputs = fc.create_model(rnn_output, scope + '_fc')
        return outputs

    def evaluation(self):
        predictions = self.get_predictions
        eval = Evaluation(self.validate_data_provider, self.batch_size, self.epochs,
                          self.num_classes, self.learning_rate, predictions)
        eval.start_evaluation()


def main():
    net = EMTAN()
    if net.operation == 'process_stats':
        net.process_stats()
    elif net.operation == 'show_stats_distribution':
        utils.stats_distribution('./data/raw/train_set.csv')
    elif net.operation == 'generate':
        if net.is_multi:
            net.tfrecords_generate()
        elif net.is_arousal:
            net.arousal_tfrecords_generate()
        elif net.is_valence:
            net.valence_tfrecords_generate()
        elif net.is_dominance:
            net.dominance_tfrecords_generate()
    elif net.operation == 'training':
        if net.is_multi:
            net.multi_task_training()
        elif net.is_arousal:
            net.arousal_training()
        elif net.is_valence:
            net.valence_tfrecords_generate()
        elif net.is_dominance:
            net.dominance_tfrecords_generate()
    elif net.operation == 'evaluation':
        if net.is_multi:
            net.multi_task_validation()
        elif net.is_arousal:
            net.arousal_validation()
        elif net.is_valence:
            net.valence_tfrecords_generate()
        elif net.is_dominance:
            net.dominance_tfrecords_generate()

if __name__ == '__main__':
    main()