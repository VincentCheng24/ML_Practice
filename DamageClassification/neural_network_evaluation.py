import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pickle
from utils import metrics
import matplotlib
matplotlib.use("TkAgg")


def neural_network_eval(model, test_x, test_y):
    """
    :param model: the neural network model
    :param test_x: test features
    :param test_y: test damages
    :return: None
    """

    y_pre_prob = model.predict(test_x)

    y_pre_cls = [np.argmax(x) for x in y_pre_prob]

    top1_acc = metrics.top1_acc(y_pre_cls, test_y)

    full_matching_acc = metrics.exact_matching_acc(y_pre_prob, test_y, threshold=0.5)

    print('top 1 precision: ', top1_acc)
    print('exact matching acc: ', full_matching_acc)


if __name__ == '__main__':
    # region
    # parameters to specify: test set, neural network model
    train_set_ratio = 0.9
    num_hidden_neurons = 100

    pickel_path = './data/data_all.pickel'
    with open(pickel_path, 'rb') as handle:
        data = pickle.load(handle)

    Y = data['damages']
    X = data['features']

    data_size = len(X)
    train_size = int(data_size * train_set_ratio)
    test_size = data_size - train_size

    test_x = X[train_size:, :]
    test_y = Y[train_size:, :]

    assert len(test_x) == test_size
    assert len(test_y) == test_size

    model = Sequential()
    model.add(Dense(num_hidden_neurons, activation='relu', input_shape=(500,)))
    model.add(Dense(5, activation='sigmoid'))

    last_saved_model = './model/weights_100_adam_lr4.hdf5'

    model.load_weights(last_saved_model, by_name=True)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # endregion

    neural_network_eval(model, test_x, test_y)


