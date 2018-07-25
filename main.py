import utils


class EMTAN():

    def __init__(self):
        self.operation = 'process_stats'

    def process_stats(self):
        original_file = './data/train.csv'
        csv_file = './data/train_set.csv'
        utils.preprocess_stats(original_file, csv_file)

def main():
    net = EMTAN()
    if net.operation == 'process_stats':
        net.process_stats()

if __name__ == '__main__':
    main()