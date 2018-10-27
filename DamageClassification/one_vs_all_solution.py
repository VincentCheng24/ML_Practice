from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras import optimizers
from parser import csv_parser


data_path = './data/data_all.csv'
X, Y = csv_parser(data_path)

train_x = X[:9000, :]
train_y = Y[:9000, :]
test_x = X[9000:, :]
test_y = Y[9000:, :]

assert len(train_x) == 9000
assert len(train_y) == 9000
assert len(test_x) == 1000
assert len(test_y) == 1000

model = Sequential()
model.add(Dense(200, activation='relu', input_shape=(500, )))
model.add(Dense(5, activation='sigmoid'))

# last_saved_model = './model/weights_adam_lr5.hdf5'
#
# model.load_weights(last_saved_model, by_name=True, skip_mismatch=True)

adam = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-7, amsgrad=False)
sgd = optimizers.SGD(lr=1e-6, decay=1e-7, momentum=0.9, nesterov=True)

algo = 'adam'
lr = 'lr4_200'
tb = TensorBoard(log_dir='../logs/{}_{}tb'.format(algo, lr), histogram_freq=0, write_grads=True, write_graph=False)
cp = ModelCheckpoint(filepath='./model/weights_{}_{}.hdf5'.format(algo, lr), verbose=1, save_best_only=True,
                     save_weights_only=False, mode='auto', period=1)
es = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=0, mode='auto')


model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x=train_x, y=train_y, batch_size=32, epochs=100, verbose=1, callbacks=[tb, cp, es],
          validation_data=[test_x, test_y], shuffle=True)
