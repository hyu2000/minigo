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


def train():
    LIMIT1, LIMIT2 = -1000, 1000
    N, N_train = 10000, 9000
    x = np.random.randint(LIMIT1, LIMIT2, (N, 2))
    y = np.min(x, axis=1)
    x_train, x_val = x[:N_train, :], x[N_train:, :]
    y_train, y_val = y[:N_train], y[N_train:]
    model = nn2(4)
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32,
                        epochs=5)
    print(history.history)


if __name__ == '__main__':
    train()
