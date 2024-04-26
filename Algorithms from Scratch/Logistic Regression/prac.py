import numpy as np

def sigmoid(x):
    return 1/ (1+np.exp(-x))

class LogisticRegression():
    def __init__(self, lr=0.01,n_iter=1000):
        self.lr=lr
        self.weights=None
        self.bias=None
        self.n_iter=n_iter
    

    def fit(self, x,y):

        n_sample,n_features=np.shape(x)

        self.bias=0
        self.weights=np.zeros(n_features)

        for _ in range(self.n_iter):
            linear_pred=np.dot(x,self.weights)+self.bias
            prediction=sigmoid(linear_pred)

            dw=(1/n_sample)*np.dot(x.T,prediction-y)
            db=(1/n_sample)*np.sum(prediction-y)

            self.weights=self.weights- self.lr*dw
            self.bias=self.bias-self.lr*db

    def predict(self, x):
        linear_pred=np.dot(x,self.weights)+self.bias
        y_pred=sigmoid(linear_pred)
        f_pred=[0 if y<=5 else 1 for y in y_pred]

        return f_pred

