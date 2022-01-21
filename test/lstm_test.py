import tensorflow as tf
import pandas as pd
from data import StockData
import evaluate


def lstm_test():
    TIME_STEPS = 30
    STOCK_TICKER = 'GOOG'
    STOCK_START_DATE = pd.to_datetime('2015-08-07')
    STOCK_VALIDATION_DATE = pd.to_datetime('2021-09-01')
    data = StockData(ticker=STOCK_TICKER, start_date=STOCK_START_DATE, validation_date=STOCK_VALIDATION_DATE)
    x_train, y_train = data.get_test_data(TIME_STEPS)
    model = tf.keras.models.load_model("../checkpoint/lstm")
    # test = min_max_scaler(df_test)
    # test_data, test_labels = ds.create_data_label(test)
    # acc = model.evaluate(x_test, y_test, verbose=2)
    # print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))
    test_predictions = model.predict(x_train)
    print("predict shape")
    print(test_predictions.shape)
    # evaluate.visualize_predict(y_train, test_predictions, model_name="LSTM")


if __name__ == '__main__':
    lstm_test()
