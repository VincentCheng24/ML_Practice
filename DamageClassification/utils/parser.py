import csv
import numpy as np
import pickle


def csv_parser(data_path):
    """
    to parse the merged csv files and perform one-hot encoding to both damages and features
    :param data_path: the path of a csv file
    :return: two np array: features and damages
    """

    damages = []
    features = []

    with open(data_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            damage = np.zeros((5, ))
            feature = np.zeros((500, ))
            damage_idx = [int(x)-1 for x in row['damage'].split(', ')]
            feature_idx = [int(x)-1 for x in row['feature'].split(', ')]
            damage[damage_idx] = 1
            feature[feature_idx] = 1
            damages.append(damage)
            features.append(feature)

    damages = np.array(damages, dtype=int)
    features = np.array(features, dtype=int)

    print('successfully parsed and encoded {}'.format(data_path))
    return features, damages


if __name__ == '__main__':
    data_path = '../data/data_all.csv'
    X, Y = csv_parser(data_path)

    # save the features and damages after one-hot encoding in dictionary format
    data = {'features': X, 'damages': Y}
    with open('../data/data_all.pickel', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('well done')
