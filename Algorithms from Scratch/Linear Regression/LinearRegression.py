import numpy as np

class LinearRegression():
    def __init__(self,lr=0.001,n_iter=2000):
        self.lr=lr
        self.n_iter=n_iter
        self.weights=None
        self.bias=None

    
    def fit(self,x,y):
        n_sample,n_feature=np.shape(x)

        self.weights=np.zeros(n_feature)
        self.bias=0

        for _ in range(n_sample):
            Y_pred=np.dot(x,self.weights)+self.bias

            dw=(1/ n_sample)*np.dot(x.T,(Y_pred-y))
            db=(1/n_sample)*np.sum(Y_pred-y)

            self.weights=self.weights-self.lr*dw
            self.bias=self.bias-self.lr*db


    def predict(self,y):
        y_pred=np.dot(y,self.weights)+self.bias

        return y_pred

  