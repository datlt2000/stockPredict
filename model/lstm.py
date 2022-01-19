import tensorflow as tf


def LstmModel(
        input_size,
        output_size,
        # size_layer=10,
        # num_layers=3,
        # learning_rate=1,
        # forget_bias=0.1,
        optimizer="adam",
        loss=tf.keras.losses.MeanSquaredError()
):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units=output_size, return_sequences=True,
                                   input_shape=(input_size, 1)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=output_size, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=output_size, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=1))
    model.summary()
    model.compile(optimizer=optimizer, loss=loss, metrics=[
        tf.keras.metrics.MeanSquaredError(name='MSE')
    ])
    return model
