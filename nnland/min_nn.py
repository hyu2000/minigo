""" DNN to compute min(a, b)
"""
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


def nn2(num_hidden_nodes):
    """ """
    inputs = keras.Input(shape=(2,))
    x = keras.layers.Dense(num_hidden_nodes, activation='relu')(inputs)
    output = keras.layers.Dense(1, activation=None)(x)
    model = keras.Model(inputs, output)

    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt, loss='mse',
                  metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return model


def gen_data1(LIMIT1, LIMIT2):
    N = 10000
    x = np.random.randint(LIMIT1, LIMIT2, (N, 2))
    y = np.min(x, axis=1)
    IDX_VAL0 = -1000
    x_train, x_val = x[:IDX_VAL0, :], x[IDX_VAL0:, :]
    y_train, y_val = y[:IDX_VAL0], y[IDX_VAL0:]
    return (x_train, y_train), (x_val, y_val)


def train():
    model = nn2(4)
    data_train, data_val = gen_data1(0, 1000)
    history = model.fit(data_train[0], data_train[1], validation_data=data_val, batch_size=32, epochs=100)
    print(history.history)


def test():
    """ generalization analysis """
    model = keras.models.load_model('model3.ok.h5')
    model.summary()
    data = np.array([[300, -2000], [-300, 2000], [-300, 5000], [-3000, 5000]])
    y_hat = model.predict(data)
    print(y_hat)


if __name__ == '__main__':
    test()
