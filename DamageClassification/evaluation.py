import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from parser import csv_parser


def top1_acc(y_pre, y_gt):
    """
    :param y_pre: 1D np array
    :param y_gt: 2D np array
    :return: float, acc
    """
    cnt = 0
    for i, val in enumerate(y_pre):
        if y_gt[i, val] == 1:
            cnt += 1

    return cnt/len(y_pre)


def exact_matching_acc(y_pre_prob, y_gt, threshold=0.5):
    """
    :param y_pre_prob: 1D np array
    :param y_gt:       2D np array
    :param threshold:  [0,1] to assgin class
    :return:  float, acc
    """
    y_pre = y_pre_prob > threshold
    cnt = 0
    for i in range(len(y_pre_prob)):
        if all(y_pre[i, :] == y_gt[i, :]):
            cnt += 1

    return cnt/len(y_pre_prob)


if __name__ == '__main__':

    data_path = './data/data_all.csv'
    X, Y = csv_parser(data_path)

    test_x = X[9000:, :]
    test_y = Y[9000:, :]

    assert len(test_x) == 1000
    assert len(test_y) == 1000

    model = Sequential()
    model.add(Dense(200, activation='relu', input_shape=(500, )))
    model.add(Dense(5, activation='sigmoid'))

    last_saved_model = './model/weights_adam_lr4_200.hdf5'

    model.load_weights(last_saved_model, by_name=True)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    y_pre_prob = model.predict(test_x)

    y_pre = [np.argmax(x) for x in y_pre_prob]

    top1_acc = top1_acc(y_pre, test_y)
    full_matching_acc = exact_matching_acc(y_pre_prob, test_y, threshold=0.5)
    print('top 1 precision: ', top1_acc)
    print('exact matching acc: ', full_matching_acc)


