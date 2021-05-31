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
    LIMIT1, LIMIT2 = 0, 1000
    N = 10000
    x = np.random.randint(LIMIT1, LIMIT2, (N, 2))
    y = np.min(x, axis=1)
    IDX_VAL0 = -1000
    x_train, x_val = x[:IDX_VAL0, :], x[IDX_VAL0:, :]
    y_train, y_val = y[:IDX_VAL0], y[IDX_VAL0:]
    model = nn2(4)
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=100)
    print(history.history)


if __name__ == '__main__':
    train()
