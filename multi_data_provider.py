import tensorflow as tf


class MultiDataProvider():

    def __init__(self, tfrecords_file, batch_size, is_shuffle):
        self.tfrecords_file = tfrecords_file
        self.batch_size = batch_size
        self.is_shuffle = is_shuffle

    def get_batch(self):
        dataset = tf.data.TFRecordDataset(self.tfrecords_file)
        dataset = dataset.map(self.parse_example)
        if self.is_shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        padded_shapes = ([None], [1], [1], [1], [])
        self.dataset = dataset.padded_batch(self.batch_size, padded_shapes=padded_shapes)

    def parse_example(self, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'file': tf.FixedLenFeature([], tf.string),
                'arousal': tf.FixedLenFeature([], tf.string),
                'valence': tf.FixedLenFeature([], tf.string),
                'dominance': tf.FixedLenFeature([], tf.string),
                'frame': tf.FixedLenFeature([], tf.string)
            }
        )

        frame = features['frame']
        arousal = features['arousal']
        valence = features['valence']
        dominance = features['dominance']
        file = features['file']

        frame = tf.decode_raw(frame, tf.float32)
        arousal = tf.decode_raw(arousal, tf.int32)
        valence = tf.decode_raw(valence, tf.int32)
        dominance = tf.decode_raw(dominance, tf.int32)

        return frame, arousal, valence, dominance, file