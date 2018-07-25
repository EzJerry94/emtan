import utils
from tfrecord_generator import Generator

class EMTAN():

    def __init__(self):
        self.operation = 'generate'

    def process_stats(self):
        original_file = './data/train.csv'
        csv_file = './data/train_set.csv'
        utils.preprocess_stats(original_file, csv_file)

    def tfrecords_generate(self):
        generator = Generator()
        generator.write_tfrecords()

def main():
    net = EMTAN()
    if net.operation == 'process_stats':
        #net.process_stats()
        utils.stats_distribution('./data/train_set.csv')
    elif net.operation == 'generate':
        net.tfrecords_generate()

if __name__ == '__main__':
    main()