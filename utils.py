import numpy as np


def preprocess_stats(file, csv_name):
    lines = np.loadtxt(file, dtype='str', usecols=(0,2,3,4))
    for sample in lines:
        #arousal; valence; dominance
        arousal = sample[1]
        valence = sample[2]
        dominance = sample[3]
        sample[1] = change_stats_to_int(arousal)
        sample[2] = change_stats_to_int(valence)
        sample[3] = change_stats_to_int(dominance)
    lines[:, 1:].astype(int)
    np.savetxt('./data/'+csv_name, lines, fmt='%s %s %s %s')

def change_stats_to_int(attribute):
    if attribute == 'pos':
        attribute = 0
    elif attribute == 'neg':
        attribute = 1
    elif attribute == 'neu':
        attribute = 2
    return attribute