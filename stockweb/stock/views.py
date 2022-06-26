from django.shortcuts import render
from django.http import JsonResponse
import json
import yfinance as yf
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np

MEAN = 10
PTC = True
TIME_STEPS = 32
CHECKPOINT_FOLDER = "../checkpoint/"
tf_models = dict()


def index(request):
    return render(request, "stock/index.html")


@csrf_exempt
def filter_data(request):
    if request.method == 'POST':
        field = json.loads(request.body)
        ticker = yf.Ticker(field['ticker'])
        data = ticker.history(interval='1d', start=field['start'], end=field['end'])
        predict_data = predict(field['model'], field['ticker'], field['start'], field['end'])
        result = {"real": data.to_csv(header=False), "predict": predict_data.to_csv(header=False)}
        return JsonResponse(result, safe=False)

    return JsonResponse("FAILED TO FILTER!")


def predict(model, ticker, start, end):
    n_ticker = yf.Ticker(ticker)
    test_data = n_ticker.history(interval="1d", start=start, end=end)
    test_data[['Close']] = test_data[['Close']].rolling(MEAN).mean()
    test_data.dropna(how='any', axis=0, inplace=True)  # Drop all rows with NaN values
    base_value = test_data['Close'].iloc[TIME_STEPS]
    test_data['ptc_change'] = test_data['Close'].pct_change()
    test_data.dropna(how='any', axis=0, inplace=True)  # Drop all rows with NaN values
    # test_data['Close'].plot(title="Close Price")
    x_test, y_test, min_max_test = preprocessing(test_data)

    if model not in tf_models:
        tf_models[model] = tf.keras.models.load_model(CHECKPOINT_FOLDER + "lstm")
    test_predictions = tf_models[model].predict(x_test)
    # print("predict shape")
    # print(test_predictions.shape)
    # visualize.visualize_predict(y_test,
    #     test_predictions, model_name="LSTM")
    # test_eval = model.evaluate(x_test, y_test, verbose=0)
    # print('Test Data - Loss: {:.4f}, MSE: {:.4f}'.format(test_eval[0], test_eval[1]))
    test_predictions = min_max_test.inverse_transform(test_predictions)
    test_predictions = pd.DataFrame(test_predictions)
    test_predictions.rename(
        columns={0: 'ptc_predict'}, inplace=True)
    # test_predictions = test_predictions.round(decimals=0)
    test_predictions.index = test_data[TIME_STEPS:].index
    # visualize.visualize_predict(test_data['Close'],
    #                             test_predictions, model_name="LSTM")
    test_data["ptc_change_predict"] = test_predictions['ptc_predict']
    test_data['close_meaning_predict'] = test_predictions['ptc_predict'].add(1, fill_value=0).cumprod() * base_value
    return test_data[['Close', 'close_meaning_predict', 'ptc_change', 'ptc_change_predict']].copy()


def preprocessing(test_data):
    """Normalize price columns"""
    min_max_test = MinMaxScaler()
    min_max_test.fit(test_data[['ptc_change']])
    test_scaled = min_max_test.transform(test_data[['ptc_change']])

    # Serialize data and generate labels for train data
    x_test, y_test = create_data_label(test_scaled, TIME_STEPS)
    # print('Shape Of X_Test Data :')
    # print(x_test.shape)
    return x_test, y_test, min_max_test


def create_data_label(data_train, step=60):
    """ Serialize data by slide window and generate labels after each time step
        default timesteps is 60 day
    """
    upper = data_train.shape[0]
    if step >= upper:
        print("step is greater than data")
        return
    x_train = []
    y_train = []
    for i in range(step, upper):
        x_train.append(data_train[i - step:i])
        y_train.append(data_train[i, 0])  # (Close price) is label
    x_train, y_train = np.array(x_train), np.array(y_train)
    return x_train, y_train
