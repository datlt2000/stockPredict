import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta
import random


def create_data_label(data_train, step=60):
    upper = data_train.shape[0]
    if step >= upper:
        print("step is greater than data")
        return
    x_train = []
    y_train = []
    for i in range(step, upper):
        x_train.append(data_train[i - step:i])
        y_train.append(data_train[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train


class StockData:
    def __init__(self, ticker, start_date, validation_date):
        self.start_date = start_date
        self.validation_date = validation_date
        self.ticker = ticker
        self._sec = yf.Ticker(ticker)
        self._min_max = MinMaxScaler(feature_range=(0, 1))

    def get_stock_currency(self):
        return self._sec.info['currency']

    def get_stock_short_name(self):
        return self._sec.info['shortName']

    # create label for sequence data
    # param:
    #   data_train - numpy.ndarray (Nx1)
    #   step - int number of continuous value for 1 label

    def download_transform_to_numpy(self, time_steps):
        end_date = datetime.today()
        print('End Date: ' + end_date.strftime("%Y-%m-%d"))
        data = yf.download([self.ticker], start=self.start_date, end=end_date)[['Close']]
        data = data.reset_index()
        # data.to_csv(os.path.join("./dataset", + self.ticker + '.csv'))
        # print(data)

        training_data = data[data['Date'] < self.validation_date].copy()
        test_data = data[data['Date'] >= self.validation_date].copy()
        training_data = training_data.set_index('Date')
        # Set the data frame index using column Date
        test_data = test_data.set_index('Date')
        # print(test_data)

        train_scaled = self._min_max.fit_transform(training_data)

        # Training Data Transformation
        x_train, y_train = create_data_label(train_scaled, time_steps)

        total_data = pd.concat((training_data, test_data), axis=0)
        inputs = total_data[len(total_data) - len(test_data) - time_steps:]
        test_scaled = self._min_max.fit_transform(inputs)

        # Testing Data Transformation
        x_test, y_test = create_data_label(test_scaled, time_steps)

        return (x_train, y_train), (x_test, y_test), (training_data, test_data)

    @staticmethod
    def __date_range(start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    @staticmethod
    def negative_positive_random():
        return 1 if random.random() < 0.5 else -1

    @staticmethod
    def pseudo_random():
        return random.uniform(0.01, 0.03)

    def generate_future_data(self, time_steps, min_max, start_date, end_date, latest_close_price):
        x_future = []
        y_future = []

        # We need to provide a randomisation algorithm for the close price
        # This is my own implementation and it will provide a variation of the
        # close price for a +-1-3% of the original value, when the value wants to go below
        # zero, it will be forced to go up.

        original_price = latest_close_price

        for single_date in self.__date_range(start_date, end_date):
            x_future.append(single_date)
            direction = self.negative_positive_random()
            random_slope = direction * (self.pseudo_random())
            # print(random_slope)
            original_price = original_price + (original_price * random_slope)
            # print(original_price)
            if original_price < 0:
                original_price = 0
            y_future.append(original_price)

        test_data = pd.DataFrame({'Date': x_future, 'Close': y_future})
        test_data = test_data.set_index('Date')

        test_scaled = min_max.fit_transform(test_data)
        x_test = []
        y_test = []
        # print(test_scaled.shape[0])
        for i in range(time_steps, test_scaled.shape[0]):
            x_test.append(test_scaled[i - time_steps:i])
            y_test.append(test_scaled[i, 0])
            # print(i - time_steps)

        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return x_test, y_test, test_data


def split_data(data, test_size=0.2):
    if data is None:
        print("you do not set data")
    else:
        sample_data = data.sample(frac=1)
        split_index = int(round(test_size * len(data), 0))
        df_train = sample_data.iloc[:-split_index]
        df_test = sample_data.iloc[-split_index:]
        return df_train, df_test


def read_data(file_name='GOOG.csv'):
    file_path = "../../dataset/" + file_name
    df = pd.read_csv(file_path)
    print(df.head())
    return df


def load_data(file_name):
    df = read_data(file_name)
    data = df.iloc[:, 4:5].astype('float32')  # Close price
    df_train, df_test = split_data(data)
    return df_train, df_test
