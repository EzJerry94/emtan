import utils
import argparse
import os


class Emtan:

    def __init__(self):
        pass


def main(args):
    print(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data/',
                        help='path to the data directory')
    parser.add_argument('--data_name',
                        type=str,
                        choices=['cried', 'iemocap'],
                        help='name of the dataset')
    parser.add_argument('--mode',
                        choices=['generate', 'train', 'test'])
    parser.add_argument('--gpu',
                        type=str,
                        default='-1')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    main(args)