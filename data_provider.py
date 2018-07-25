import tensorflow as tf


class DataProvider():

    def __init__(self, tfrecords_file, batch_size, is_shuffle):
        self.tfrecords_file = tfrecords_file
        self.batch_size = batch_size
        self.is_shuffle = is_shuffle

    def get_batch(self):
        dataset = tf.data.TFRecordDataset(self.tfrecords_file)
        dataset = dataset.map(self.parse_example)
        if self.is_shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        padded_shapes = ([None], [1])
        self.dataset = dataset.padded_batch(self.batch_size, padded_shapes=padded_shapes)

    def parse_example(self, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'file': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
                'frame': tf.FixedLenFeature([], tf.string),
            }
        )

        frame = features['frame']
        label = features['label']

        frame = tf.decode_raw(frame, tf.float32)
        label = tf.decode_raw(label, tf.int32)

        return frame, label