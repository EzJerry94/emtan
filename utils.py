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

def stats_distribution(csv):
    lines = np.loadtxt(csv, dtype='str')
    first_neu = 0
    first_pos = 0
    first_neg = 0
    second_neu = 0
    second_pos = 0
    second_neg = 0
    third_neu = 0
    third_pos = 0
    third_neg = 0
    for line in lines:
        first = line[1]
        if first == '0':
            first_neu += 1
        elif first == '1':
            first_pos += 1
        else:
            first_neg += 1

        second = line[2]
        if second == '0':
            second_neu += 1
        elif second == '1':
            second_pos += 1
        else:
            second_neg += 1

        third = line[3]
        if third == '0':
            third_neu += 1
        elif third == '1':
            third_pos += 1
        else:
            third_neg += 1
    print('arousal_neu:', first_neu)
    print('arousal_pos:', first_pos)
    print('arousal_neg:', first_neg)
    print('valence_neu:', second_neu)
    print('valence_pos:', second_pos)
    print('valence_neg:', second_neg)
    print('dominance_neu:', third_neu)
    print('dominance_pos:', third_pos)
    print('dominance_neg:', third_neg)
