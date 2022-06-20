from sklearn.preprocessing import MinMaxScaler


# normalization data
# param:
#   df - like pandas.DataFrame
# return - like numpy.ndarray
def min_max_scaler(df):
    if df is None:
        print("forgot data to normalization")
    else:
        try:
            minmax = MinMaxScaler(feature_range=(0, 1))
            df_train = minmax.fit_transform(df)  # Close index
            return df_train
        except:
            print("cannot normalization")
