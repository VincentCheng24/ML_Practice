import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from utils import metrics
import matplotlib
matplotlib.use("TkAgg")


def training():
    # specify the parameters
    train_set_ratio = 0.9

    # load the data and split it into training and test set w.r.t. training_set_ratio
    pickel_path = './data/data_all.pickel'
    with open(pickel_path, 'rb') as handle:
        data = pickle.load(handle)

    Y = data['damages']
    X = data['features']

    data_size = len(X)
    train_size = int(data_size * train_set_ratio)
    test_size = data_size - train_size

    train_x = X[:train_size, :]
    train_y = Y[:train_size, :]
    test_x = X[train_size:, :]
    test_y = Y[train_size:, :]

    assert len(train_x) == train_size
    assert len(train_y) == train_size
    assert len(test_x) == test_size
    assert len(test_y) == test_size

    # Using loop for applying logistic regression and one vs rest classifier
    categories = [1, 2, 3, 4, 5]
    predictions = []

    for category in categories:
        print('Processing Damage {}'.format(category))

        # build the model
        model = Sequential()
        model.add(Dense(1, activation='sigmoid', input_shape=(500, ),
                        kernel_initializer='random_uniform', bias_initializer='zeros', name='model_{}'.format(category)))

        es = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=0, mode='auto')
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Training logistic regression model on train data
        model.fit(x=train_x, y=train_y[:, category-1], batch_size=32, epochs=100, verbose=1, callbacks=[es],
                  validation_data=[test_x, test_y[:, category-1]], shuffle=True)

        # prediction on the test set
        prediction = model.predict(test_x)
        predictions.append(prediction)

    # evaluation
    y_pre_prob = np.transpose(np.array(predictions))[0]
    y_pre_cls = [np.argmax(x) for x in y_pre_prob]

    # two metrics
    top1_acc = metrics.top1_acc(y_pre_cls, test_y)
    full_matching_acc = metrics.exact_matching_acc(y_pre_prob, test_y, threshold=0.5)

    print('top 1 precision: ', top1_acc)
    print('exact matching acc: ', full_matching_acc)


if __name__ == '__main__':
    training()

