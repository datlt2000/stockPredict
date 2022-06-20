import numpy as np
import matplotlib.pyplot as plt


def calculate_accuracy(real, predict):
    real = np.array(real) + 1
    predict = np.array(predict) + 1
    percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
    return percentage * 100

    
def visualize_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss/Validation Loss')
    plt.legend(loc='upper right')
    plt.pause(0.001)
    plt.show()
    
def visualize_predict(real_data, predict_data, model_name="LSTM"):
    plt.figure(figsize=(16,6))
    plt.plot(real_data, label='True Value')
    plt.plot(predict_data, label='Predict Value')
    plt.title('Prediction by' + model_name)
    plt.xlabel('Time Scale')
    plt.ylabel('Scaled USD')
    plt.legend()
    plt.show()