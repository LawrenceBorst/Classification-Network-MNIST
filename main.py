import numpy as np
import NeuralNet
import LoadData

data = LoadData.LoadData(X_labels='samples/train-labels-idx1-ubyte',
                         X_images='samples/train-images-idx3-ubyte',
                         Y_labels='samples/t10k-labels-idx1-ubyte',
                         Y_images='samples/t10k-images-idx3-ubyte')  # Load the data
X_labels, X_images = data.get_train_data()
X_images = data.preprocess(X_images)

neural = NeuralNet.NeuralNet()

for i in range(0, 10000):
    if i % 500 == 0:
        print(i)
    # Initialize Y to a vector of 1s
    Y = np.zeros(10)
    Y[X_labels[i]] = 1.0

    neural.set_Y(Y)
    neural.set_X(X_images[i])
    neural.forward()
    neural.back_propagate()

Y_labels, Y_images = data.get_test_data()
Y_images = data.preprocess(Y_images)

count = 0
for i in range(0, 10000):
    if i % 500 == 0:
        print(i)
    # Initialize Y to a vector of 1s
    Y = np.zeros(10)
    Y[Y_labels[i]] = 1.0

    neural.set_Y(Y)
    neural.set_X(Y_images[i])
    neural.forward()
    if neural.get_results() == Y_labels[i]:
        count += 1
print(count)
