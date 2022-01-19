import data as ds
import tensorflow as tf
from normalization import min_max_scaler
import pandas as pd
from data import StockData


def lstm_test():
    TIME_STEPS = 60
    STOCK_TICKER = 'GOOG'
    STOCK_START_DATE = pd.to_datetime('2015-08-07')
    STOCK_VALIDATION_DATE = pd.to_datetime('2021-09-01')
    data = StockData(ticker=STOCK_TICKER, start_date=STOCK_START_DATE, validation_date=STOCK_VALIDATION_DATE)
    (x_train, y_train), (x_test, y_test), (training_data, test_data) = data.download_transform_to_numpy(TIME_STEPS)
    model = tf.keras.models.load_model("../checkpoint/lstm")
    # test = min_max_scaler(df_test)
    # test_data, test_labels = ds.create_data_label(test)
    acc = model.evaluate(x_test, y_test, verbose=2)
    print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))
    test_predictions_baseline = model.predict(x_test)


if __name__ == '__main__':
    lstm_test()
