import numpy as np

class NeuralNet:
    def __init__(self, X_labels, X_images):
        self.X_labels = X_labels
        self.X_images = X_images

        self.X = np.array([0.1, 0.1, 0.1])
        self.W1 = np.array([[0.3, -0.4, 0.1], [0.2, 0.6, 0.1], [0.1, -0.2, -0.1]])
        self.B1 = np.array([0.25, 0.45, 0.1])
        self.W2 = np.array([[0.7, 0.5, 0.2], [-0.3, -0.1, 0.2], [0.1, 0.1, 0.1], [0.2, 0.2, 0.3]])
        self.B2 = np.array([0.15, 0.35, 0.25, 0.1])
        self.H1 = np.array([0, 0, 0])
        self.O = np.array([0.2, 1, 2, 1])
        self.Y = np.array([0.2, 1, 1, 1])

        self.X_size = len(self.X)
        self.H1_size = len(self.H1)
        self.O_size = len(self.O)

    def act_func(self, arr):
        def f(x):
            return 1 / (1 + np.exp(-x))
        return f(arr)

    def grad_act_func(self, x):
        x = x * (1 - x)
        return x

    def loss_func(self, x):
        loss = 0
        for i in range(0, 2):
            loss += (self.O[i] - self.Y[i])**2

    def grad_loss_func(self, x, j):
        return 2 * (x - self.Y[j])

    def forward(self):
        self.H1 = self.act_func(np.dot(self.W1, self.X) + self.B1)
        self.O = self.act_func(np.dot(self.W2, self.H1) + self.B2)

    def descend(self):  # This was painstakingly turned into a vector equation
                        # Based partly on the article "gradient descent and backpropagation in towardsdatascience.com"
        db2 = []

        for i in range(self.O_size):
            db2.append(self.grad_act_func(self.O[i]) * self.grad_loss_func(self.O[i], i))

        X_matrix = self.X.reshape(self.X_size, 1)
        H1_matrix = self.H1.reshape(self.H1_size, 1)
        db2 = np.array(db2)
        helper_matrix = db2.reshape(1, self.O_size)
        dw2 = (np.matmul(H1_matrix, helper_matrix)).T

        x = np.tile(np.matmul(helper_matrix, self.W2).T, self.X_size)
        y = np.matmul(X_matrix, self.grad_act_func(H1_matrix.T))
        dw1 = np.multiply(x.T, y).T
        db1 = np.multiply(np.matmul(helper_matrix, self.W2), self.grad_act_func(H1_matrix.T)).reshape(self.H1_size)

        return dw1, dw2, db1, db2

    def back_propagate(self):
        dw1, dw2, db1, db2 = self.descend()
        self.W2 -= dw2 * 0.1
        self.W1 -= dw1 * 0.1
        self.B2 -= db2 * 0.1
        self.B1 -= db1 * 0.1









    """def __relu(self, arr):
        def f(x):
            if x <0:
                return 0
            else:
                return x
        return f(arr)

    def __grad_loss(self, scores):
        loss = 0
        # Naive loss
        for i in range(0, 10):  # Per Image
            L_i = 0
            for j in range(0, 10):  # Computation of Loss
                if j != self.X_labels[i]:
                    L_i += self.__relu(scores[j] - self.X_labels[i] + 1)
            loss += L_i / 60000     # 60000 is the number of images
        # Regularization
        reg_loss = 1.0 * np.sum(np.multiply(self.W1, self.W1)) + np.sum(np.multiply(self.W2, self.W2))
        loss += reg_loss
        return loss

    def back_propagate(self):
        scores = []
        for i in range(0, 10):                                          # Per image
            H1 = self.__forward(self.X_images[i], self.W1, self.B1)     # Hidden layers
            score = self.__forward(self.H2, self.W2, self.B2)      # 10-node output layer with digit probabilities
            scores.append(score)"""











"""
import numpy as np

class NeuralNet:
    def __init__(self):
        self.X = np.zeros(28 * 28)
        self.Y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.W1 = 1 / 60000 * np.random.rand(200, 784)
        self.B1 = np.random.rand(200)
        self.W2 = 1 / 60000 * np.random.rand(10, 200)
        self.B2 = np.random.rand(10)
        self.H1 = np.random.rand(200)
        self.O = np.zeros(10, dtype="float32")

        self.X_size = len(self.X)
        self.H1_size = len(self.H1)
        self.O_size = len(self.O)

    def set_X(self, X):
        self.X = X

    def set_Y(self, Y):
        self.Y = Y


    def act_func(self, arr):
        def f(x):
            return 1 / (1 + np.exp(-x))
        return f(arr)

    def grad_act_func(self, x):
        x = x * (1 - x)
        return x

    def loss_func(self, x):
        loss = 0
        for i in range(0, 2):
            loss += (self.O[i] - self.Y[i])**2

    def grad_loss_func(self, x, j):
        return 2 * (x - self.Y[j])

    def forward(self):
        self.H1 = self.act_func(np.dot(self.W1, self.X) + self.B1)
        self.O = self.act_func(np.dot(self.W2, self.H1) + self.B2)

    def descend(self):  # This was painstakingly turned into a vector equation
                        # Based partly on the article "gradient descent and backpropagation in towardsdatascience.com"
        db2 = []

        for i in range(self.O_size):
            db2.append(self.grad_act_func(self.O[i]) * self.grad_loss_func(self.O[i], i))

        X_matrix = self.X.reshape(self.X_size, 1)
        H1_matrix = self.H1.reshape(self.H1_size, 1)
        db2 = np.array(db2)
        helper_matrix = db2.reshape(1, self.O_size)
        dw2 = (np.matmul(H1_matrix, helper_matrix)).T

        x = np.tile(np.matmul(helper_matrix, self.W2).T, self.X_size)
        y = np.matmul(X_matrix, self.grad_act_func(H1_matrix.T))
        dw1 = np.multiply(x.T, y).T
        db1 = np.multiply(np.matmul(helper_matrix, self.W2), self.grad_act_func(H1_matrix.T)).reshape(self.H1_size)

        return dw1, dw2, db1, db2

    def back_propagate(self):
        dw1, dw2, db1, db2 = self.descend()
        self.W2 -= dw2 * 0.1
        self.W1 -= dw1 * 0.1
        self.B2 -= db2 * 0.1
        self.B1 -= db1 * 0.1

    def get_results(self):
        return np.argmax(self.O)



"""