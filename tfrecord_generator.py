import tensorflow as tf
import numpy as np
import copy
import random
from moviepy.editor import AudioFileClip


class Generator:

    def __init__(self):
        self.csv = 'data/raw/train_set.csv'
        self.upsample = True
        self.classes = 3
        self.tfrecords_file = 'data/arousal/multi_train_set.tfrecords'
        self.attribute = 'arousal'
        self.num_arousal_after_upsample = 9051
        self.num_valence_after_upsample = 14022
        self.num_dominance_after_upsample = 12393

    def _int_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def read_csv(self, file, attribute):
        if attribute == 'arousal':
            self.data = np.loadtxt(file, dtype='str', usecols=(0, 1))
        elif attribute == 'valence':
            self.data = np.loadtxt(file, dtype='str', usecols=(0, 2))
        elif attribute == 'dominance':
            self.data = np.loadtxt(file, dtype='str', usecols=(0, 3))

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

    def upsample_process(self, sample_data):
        classes = [int(x['label']) for x in sample_data.values()]
        class_ids = set(classes)
        num_samples_per_class = {class_name: sum(x == class_name for x in classes) for class_name in class_ids}

        max_samples = np.max(list(num_samples_per_class.values()))
        augmented_data = copy.copy(sample_data)
        for class_name, n_samples in num_samples_per_class.items():
            n_samples_to_add = max_samples - n_samples

            while n_samples_to_add > 0:
                for key, value in sample_data.items():
                    label = int(value['label'])
                    sample = key
                    if n_samples_to_add <= 0:
                        break

                    if label == class_name:
                        augmented_data[sample + '_' + str(n_samples_to_add)] = {'file': value['file'],
                                                                                'label': np.int32(label)}
                        n_samples_to_add -= 1

        return augmented_data

    def multi_sample_num_same(self, data, diff):
        each_class = diff // 3
        neu_index = [] # 0
        pos_index = [] # 1
        neg_index = [] # 2
        for item in data.keys():
            if data[item]['label'] == 0:
                neu_index.append(item)
            elif data[item]['label'] == 1:
                pos_index.append(item)
            else:
                neg_index.append(item)

        random.seed(3)
        neu_add_index = random.sample(neu_index, each_class)
        pos_add_index = random.sample(pos_index, each_class)
        neg_add_index = random.sample(neg_index, each_class)
        for index in neu_add_index:
            data[index + '_' + 'multi'] = {'file': data[index]['file'],
                                           'label': data[index]['label']}
        for index in pos_add_index:
            data[index + '_' + 'multi'] = {'file': data[index]['file'],
                                           'label': data[index]['label']}
        for index in neg_add_index:
            data[index + '_' + 'multi'] = {'file': data[index]['file'],
                                           'label': data[index]['label']}
        return data

    def write_tfrecords(self):
        self.read_csv(self.csv, self.attribute)
        self.dict_files = dict()
        for row in self.data:
            self.dict_files[row[0]] = {'file': row[0],
                                       'label': np.int32(row[1]),
                                       }
        if self.upsample:
            self.dict_files = self.upsample_process(self.dict_files)

        if self.attribute == 'arousal':
            self.dict_files = self.multi_sample_num_same(self.dict_files,
                                                         self.num_valence_after_upsample - self.num_arousal_after_upsample)
        if self.attribute == 'dominance':
            self.dict_files = self.multi_sample_num_same(self.dict_files,
                                                         self.num_valence_after_upsample - self.num_dominance_after_upsample)

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
                'label': self._bytes_feature(self.dict_files[data_file]['label'].tobytes()),
                'frame': self._bytes_feature(frame.tobytes())
            }))
            writer.write(example.SerializeToString())

            index += 1

        writer.close()