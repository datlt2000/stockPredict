import numpy as np
import matplotlib.pyplot as plt


def calculate_accuracy(real, predict):
    real = np.array(real) + 1
    predict = np.array(predict) + 1
    percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
    return percentage * 100


def visualize_loss(model_out):
    if model_out is None:
        print("You do not train model")
        return
    plt.plot(model_out.history['loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


def visualize_predict(real_data, predict_data, model_name="LSTM"):
    plt.plot(real_data, label='True Value')
    plt.plot(predict_data, label='Predict Value')
    plt.title('Prediction by' + model_name)
    plt.xlabel('Time Scale')
    plt.ylabel('Scaled USD')
    plt.legend()
    plt.show()
