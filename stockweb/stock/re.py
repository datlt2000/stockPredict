def moving_average(test_data):
    df_mean = test_data[FEATURE].rolling(MEAN).mean() 
    # Drop all rows with NaN values
    df_mean.dropna(how='any', axis=0, inplace=True)
    return df_mean

def percent_change(test_data):
    #percent change
    base_value = test_data['Close'].iloc[TIME_STEPS]
    test_data['Open'] = test_data['Open'].pct_change()
    test_data['High'] = test_data['High'].pct_change()
    test_data['Low'] = test_data['Low'].pct_change()
    test_data['Close'] = test_data['Close'].pct_change()
    test_data['Volume'] = test_data['Volume'].pct_change()
    # Drop all rows with NaN values
    test_data.dropna(how='any', axis=0, inplace=True)
    return test_data, base_value

def normalization(test_data):
    #Normalize price columns
    scaler = MinMaxScaler()
    scaler.fit(test_data[FEATURE])
    test_scaled = scaler.transform(test_data[FEATURE])

    # Serialize data and generate labels for train data
    x_test, y_test = create_data_label(test_scaled, TIME_STEPS)
    # print('Shape Of X_Test Data :')
    # print(x_test.shape)
    return x_test, y_test


def create_data_label(data_train, step=60):
    """ Serialize data by slide window and generate labels after each time step
        default timesteps is 60 day
        param:
          data_train - numpy.ndarray with shape (length, 5)
    """
    upper = data_train.shape[0]
    if step >= upper:
        print("step is greater than data")
        return
    x_train = []
    y_train = []
    for i in range(step, upper):
        x_train.append(data_train[i - step:i])
        y_train.append(data_train[i][3])    # (Close price) is label
    x_train, y_train = np.array(x_train), np.array(y_train)
    return x_train, y_train

def predict(model, ticker, start, end):
    n_ticker = yf.Ticker(ticker)
    test_data = n_ticker.history(interval="1d", start=start, end=end)
    df_mean = moving_average(test_data.copy())
    ptc_df, base_value = percent_change(df_mean.copy())
    min_value = ptc_df["Close"].min()
    max_value = ptc_df["Close"].max()
    x_test, y_test = normalization(ptc_df.copy())

    if model not in tf_models:
        tf_models[model] = tf.keras.models.load_model(
            CHECKPOINT_FOLDER + model)
    test_predictions = tf_models[model].predict(x_test)
    # print("predict shape")
    # print(test_predictions.shape)
    # visualize.visualize_predict(y_test,
    #     test_predictions, model_name="LSTM")
    # test_eval = model.evaluate(x_test, y_test, verbose=0)
    # print('Test Data - Loss: {:.4f}, MSE: {:.4f}'.format(test_eval[0], test_eval[1]))
    test_predictions = pd.DataFrame(test_predictions)
    test_predictions.rename(
        columns={0: 'ptc_predict'}, inplace=True)
    test_predictions['ptc_predict'] = test_predictions['ptc_predict']*(max_value - min_value) + min_value
    # test_predictions = test_predictions.round(decimals=0)
    test_predictions.index = ptc_df[TIME_STEPS:].index
    # visualize.visualize_predict(test_data['Close'],
    #                             test_predictions, model_name="LSTM")
    test_data["Close"] = df_mean["Close"]
    test_data["ptc_change_predict"] = test_predictions['ptc_predict']
    test_data["ptc_change"] = ptc_df["Close"]
    test_data['close_meaning_predict'] = test_predictions['ptc_predict'].add(
        1, fill_value=0).cumprod() * base_value
    return test_data[['Close', 'close_meaning_predict', 'ptc_change', 'ptc_change_predict']]