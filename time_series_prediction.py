import numpy as np
import pandas as pd

from keras.layers import Dense
from keras.models import Sequential


def switch_to_supported_backend():
    import matplotlib
    gui_env = ['TKAgg', 'GTKAgg', 'Qt4Agg', 'WXAgg']
    for gui in gui_env:
        try:
            matplotlib.use(gui, warn=False, force=True)
            import matplotlib.pyplot as plt
            return plt
        except:
            continue


def create_dataset(data, look_back_val=1):
    data_x = []
    data_y = []
    for i in range(len(data) - look_back_val - 1):
        data_x.append(data[i:i + look_back_val, 0])
        data_y.append(data[i + look_back_val, 0])
    return np.array(data_x), np.array(data_y)


def fit_data(train_x, train_y, test_x, test_y, look_back_val):
    model = Sequential()

    model.add(Dense(12, input_dim=look_back_val, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_x, train_y, batch_size=2, epochs=400, verbose=1)

    train_score = model.evaluate(train_x, train_y, verbose=1)
    test_score = model.evaluate(test_x, test_y, verbose=1)

    print("Test score is {0} and training score is {1}".format(np.math.sqrt(test_score), np.math.sqrt(train_score)))


def fetch_data():
    dataframe = pd.read_csv("international-airline-passengers.csv", skiprows=0, usecols=[1])
    dataset = dataframe.values
    dataset = dataset.astype('float32')

    # plot_data(dataframe)

    look_back = 3

    train_size = int(len(dataset) * 0.67)
    train = dataset[0:train_size, :]
    test = dataset[train_size:len(dataset), :]

    train_x, train_y = create_dataset(train, look_back)
    test_x, test_y = create_dataset(test, look_back)

    fit_data(train_x, train_y, test_x, test_y, look_back)


def plot_data(dataframe):
    plot = switch_to_supported_backend()
    if plot is not None:
        plot.plot(dataframe)
        plot.show()


if __name__ == '__main__':
    np.random.seed(7)
    fetch_data()
