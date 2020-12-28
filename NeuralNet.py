import numpy as np

class NeuralNet:
    def __init__(self):
        self.p = 0.1;
        self.X = np.zeros(28 * 28)
        self.Y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.W1 = 1 / 20 * np.random.rand(200, 784)
        self.B1 = np.zeros(200)
        self.W2 = 1 / 10 * np.random.rand(10, 200)
        self.B2 = np.zeros(10)
        self.H1 = np.zeros(200)
        self.O = np.zeros(10, dtype="float32")

        self.X_size = len(self.X)
        self.H1_size = len(self.H1)
        self.O_size = len(self.O)

    def set_X(self, X):
        self.X = X

    def set_Y(self, Y):
        self.Y = Y

    def act_func(self, X, W, B):
        return np.maximum(0, np.dot(W, X) + B)

    def grad_act_func(self, x):
        x = (abs(x) + x) / 2
        return x

    def grad_loss_func(self, x, j):
        return 2 * (x - self.Y[j])

    def forward(self):
        self.H1 = self.act_func(self.X, self.W1, self.B1)
        U = (np.random.rand(*self.H1.shape) < self.p) / self.p
        self.H1 *= U
        self.O = self.act_func(self.H1, self.W2, self.B2)

    def descend(self):
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
