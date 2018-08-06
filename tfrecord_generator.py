import tensorflow as tf
import numpy as np
import copy
import random
import utils
from moviepy.editor import AudioFileClip


class Generator:

    def __init__(self):
        self.csv = 'data/raw/test_set.csv'
        self.upsample = True
        self.classes = 3
        self.tfrecords_file = 'data/multi/multi_test_set.tfrecords'

    def _int_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def read_multi_csv(self, file):
        self.data = np.loadtxt(file, dtype='str')

    def get_samples(self, data_file):
        # time = self.dict_files[data_file]['time']
        audio_clip = AudioFileClip(data_file)
        clip = audio_clip.set_fps(16000)
        # num_samples = int(clip.fps * time)
        data_frame = np.array(list(clip.subclip(0).iter_frames()))
        data_frame = data_frame.mean(1)
        chunk_size = 640  # split audio file to chuncks of 40ms
        audio = np.pad(data_frame, (0, chunk_size - data_frame.shape[0] % chunk_size), 'constant')
        audio = np.reshape(audio, (-1, chunk_size)).astype(np.float32)
        return audio

    def multi_upsample(self, sample_data, attribute):
        classes = [int(x[attribute]) for x in sample_data.values()]
        class_ids = set(classes)
        num_samples_per_class = {class_name: sum(x == class_name for x in classes) for class_name in class_ids}

        max_samples = np.max(list(num_samples_per_class.values()))
        augmented_data = copy.copy(sample_data)
        for class_name, n_samples in num_samples_per_class.items():
            n_samples_to_add = max_samples - n_samples

            while n_samples_to_add > 0:
                for key, value in sample_data.items():
                    arousal = int(value['arousal'])
                    valence = int(value['valence'])
                    dominance = int(value['dominance'])
                    sample = key
                    if n_samples_to_add <= 0:
                        break

                    if attribute == 'arousal':
                        label = arousal
                    elif attribute == 'valence':
                        label = valence
                    else:
                        label = dominance

                    if label == class_name:
                        augmented_data[sample + '_' + str(n_samples_to_add)] = {'file': value['file'],
                                                                                'arousal': np.int32(arousal),
                                                                                'valence': np.int32(valence),
                                                                                'dominance': np.int32(dominance)}
                        n_samples_to_add -= 1

        return augmented_data

    def write_multi_tfrecords(self):
        self.read_multi_csv(self.csv)
        #random.seed(3)
        #np.random.shuffle(self.data)
        self.dict_files = dict()
        for row in self.data:
            self.dict_files[row[0]] = {'file': row[0],
                                       'arousal': np.int32(row[1]),
                                       'valence': np.int32(row[2]),
                                       'dominance': np.int32(row[3])}

        utils.upsample_stats_distribution(self.dict_files)
        print('***********************************')
        '''
        self.dict_files = self.multi_upsample(self.dict_files, 'arousal')
        self.dict_files = self.multi_upsample(self.dict_files, 'valence')
        self.dict_files = self.multi_upsample(self.dict_files, 'dominance')
        utils.upsample_stats_distribution(self.dict_files)
        '''

        print('\n Start generating tfrecords \n')

        writer = tf.python_io.TFRecordWriter(self.tfrecords_file)

        index = 1
        length = len(self.dict_files)

        for data_file in self.dict_files.keys():
            print('Writing file : {} {}/{}'.format(data_file, index, length))
            frame = self.get_samples(self.dict_files[data_file]['file'])
            # frame = np.array(frame)

            example = tf.train.Example(features=tf.train.Features(feature={
                'file': self._bytes_feature(self.dict_files[data_file]['file'].encode()),
                'arousal': self._bytes_feature(self.dict_files[data_file]['arousal'].tobytes()),
                'valence': self._bytes_feature(self.dict_files[data_file]['valence'].tobytes()),
                'dominance': self._bytes_feature(self.dict_files[data_file]['dominance'].tobytes()),
                'frame': self._bytes_feature(frame.tobytes())
            }))
            writer.write(example.SerializeToString())

            index += 1

        writer.close()