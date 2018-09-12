import matplotlib.pyplot as plt
import numpy as np

def visualize_lstm_crossval(accuracies, best_option, best_result):
    options = range(len(accuracies))
    print(len(accuracies))
    plt.scatter(options[0:27], accuracies[0:27], label="Adam", marker="x")
    plt.scatter(options[27: 54], accuracies[27:54], label="SGD", marker="o")
    plt.scatter(options[54:], accuracies[54:], label="Adagrad")

    plt.scatter(best_option, best_result, c="red", s=50, label ="Best Performance")
    plt.xlabel('Parameter Validation Combination')
    plt.ylabel('Accuracy')
    plt.title('Cross Validation Results')
    plt.legend(loc="best")
    plt.show()


def visualize_rnn_crossval(accuracies, best_option, best_result):
    options = range(len(accuracies))
    plt.scatter(options, accuracies)

    plt.scatter(best_option, best_result, c="red", s=50, label ="Best Performance")
    plt.xlabel('Parameter Validation Combination')
    plt.ylabel('Accuracy')
    plt.title('Cross Validation Results')
    plt.legend(loc="best")
    plt.show()

def visualize_results(train_accuracy, val_accuracy, epochs):
    plt.plot(epochs, train_accuracy, c="red", label ="Training Accuracy")
    plt.plot(epochs, val_accuracy, label = "Validation Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('RNN Model Results')
    plt.legend(loc="best")
    plt.show()


epochs = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
lstm_train_acc = [0.1880, 0.6107, 0.7664, 0.8583, 0.8937, 0.9194, 0.9295, 0.9338, 0.9463, 0.9513, 0.9576, 0.9548, 0.9654,  0.9692, 0.9712, 0.9696, 0.9751, 0.9759, 0.9825, 0.9724, 0.9852]
lstm_val_acc = [0.1989, 0.4487, 0.4767, 0.4720, 0.4924, 0.4982, 0.4912, 0.4936, 0.4988, 0.4994, 0.4912, 0.4936, 0.5023, 0.4971, 0.4895, 0.4959, 0.4895, 0.4819, 0.4924, 0.4895, 0.4877]
rnn_train_acc = [0.1246,0.4208, 0.5625,0.6384, 0.6933, 0.7217, 0.7427, 0.7622, 0.7606, 0.7851,  0.7863, 0.8007, 0.7910, 0.8015, 0.8065, 0.8073, 0.8202, 0.8248, 0.8147, 0.8318, 0.8264]
rnn_val_acc = [0.1698,0.2153, 0.2602,0.2818, 0.2935, 0.2946, 0.3016, 0.3046, 0.2940, 0.3016, 0.2935, 0.2929, 0.2882, 0.2853, 0.2935, 0.2958, 0.2929, 0.2894, 0.2970, 0.2946, 0.2993]

visualize_results(rnn_train_acc, rnn_val_acc, epochs)
