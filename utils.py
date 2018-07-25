import numpy as np


def preprocess_stats(file, csv_name):
    lines = np.loadtxt(file, dtype='str', usecols=(0,2,3,4))
    for line in lines:
        # wav path
        #line[0] = 'wav/' + line[0]
        # arousal
        line[1] = change_stat_to_int(line[1])
        line[2] = change_stat_to_int(line[2])
        line[3] = change_stat_to_int(line[3])
    np.savetxt(csv_name, lines, fmt='%s %s %s %s')

def change_stat_to_int(attribute):
    if attribute == 'neu':
        attribute = 0
    elif attribute == 'pos':
        attribute = 1
    elif attribute == 'neg':
        attribute = 2
    return attribute