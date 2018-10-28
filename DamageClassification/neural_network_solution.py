from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras import optimizers
from neural_network_evaluation import neural_network_eval
import pickle
import matplotlib
matplotlib.use("TkAgg")


def training():

    # parameters to specify
    train_set_ratio = 0.9
    num_hidden_neurons = 50

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

    # build the neural network model
    model = Sequential()
    model.add(Dense(num_hidden_neurons, activation='relu', input_shape=(500, ), kernel_initializer='random_uniform',
                    bias_initializer='zeros'))
    model.add(Dense(5, activation='sigmoid', kernel_initializer='zeros', bias_initializer='zeros'))

    # # load the pre-trained model if necessary
    # last_saved_model = './model/weights_adam_lr5.hdf5'
    # model.load_weights(last_saved_model, by_name=True, skip_mismatch=True)

    # specify the parameters for training and perform training
    algo = 'adam'
    lr = 'lr4'
    tb = TensorBoard(log_dir='./logs/{}_{}_{}_tb'.format(num_hidden_neurons, algo, lr), histogram_freq=0, write_grads=True, write_graph=False)
    cp = ModelCheckpoint(filepath='./model/weights_{}_{}_{}.hdf5'.format(num_hidden_neurons, algo, lr), verbose=1, save_best_only=True,
                         save_weights_only=False, mode='auto', period=1)
    es = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=0, mode='auto')
    adam = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-7, amsgrad=False)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x=train_x, y=train_y, batch_size=32, epochs=100, verbose=1, callbacks=[tb, cp, es],
              validation_data=[test_x, test_y], shuffle=True)

    # evaluate the model directly after training
    neural_network_eval(model, test_x, test_y)


if __name__ == '__main__':
    training()
