from model.lstm import LstmModel
import tensorflow.keras.callbacks as callback
import evaluate as evaluate
import pandas as pd
from data import StockData


# def visualize_result(self):
#     accuracies = [calculate_accuracy(self.df_test, r) for r in self.results]
#     plt.figure(figsize=(15, 5))
#     for no, r in enumerate(self.results):
#         plt.plot(r, label='forecast %d' % (no + 1))
#     plt.plot(self.df_test, label='true trend', c='black')
#     plt.legend()
#     plt.title('average accuracy: %.4f' % (np.mean(accuracies)))
#     plt.show()


def train():
    EPOCHS = 200
    BATCH_SIZE = 30
    TIME_STEPS = 60
    STOCK_TICKER = 'GOOG'
    STOCK_START_DATE = pd.to_datetime('2015-08-07')
    STOCK_VALIDATION_DATE = pd.to_datetime('2021-09-01')
    data = StockData(ticker=STOCK_TICKER, start_date=STOCK_START_DATE, validation_date=STOCK_VALIDATION_DATE)
    (x_train, y_train), (x_test, y_test), (training_data, test_data) = data.download_transform_to_numpy(TIME_STEPS)
    # df_train, df_test = ds.load_data(STOCK_TICKER)
    # train = min_max_scaler(df_train)
    # train_data, train_label = ds.create_data_label(train, step=sequence_length)
    model = LstmModel(input_size=x_train.shape[1], output_size=50)
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test),
                          callbacks=[callback.EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)])
    model.save("../checkpoint/lstm")
    evaluate.visualize_loss(history)


if __name__ == '__main__':
    train()
