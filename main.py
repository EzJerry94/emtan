import utils


class EMTAN():

    def __init__(self):
        self.operation = 'process_stats'

    def process_stats(self):
        csv_file = './data/stats.txt'
        utils.preprocess_stats(csv_file)

def main():
    net = EMTAN()
    if net.operation == 'process_stats':
        net.process_stats()

if __name__ == '__main__':
    main()