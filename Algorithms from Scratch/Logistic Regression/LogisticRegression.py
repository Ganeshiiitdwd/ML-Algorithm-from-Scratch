import numpy as np;

# defining the sigmoid function

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegresiion():
    def __init__(self,lr=0.001,n_iter=1000):
        self.lr=lr
        self.n_iter=n_iter
        self.weights=None
        self.bias=None
    
    def fit(self,x,y):
        n_sample,n_features=np.shape(x)
        self.weights=np.zeros(n_features)
        self.bias=0


        for _ in range(self.n_iter):
            linear_pred=np.dot(x,self.weights)+self.bias
            prediction=sigmoid(linear_pred)

        # x.T it is the transpose of the matrix x
            dw=(1/n_sample)*np.dot(x.T,prediction-y)
            db=(1/n_sample)*np.sum(prediction-y)


            self.weights=self.weights-self.lr*dw
            self.bias=self.bias-self.lr*db
        # print("Final weights:", self.weights)
        # print("Final bias:", self.bias)
        

    
    def predict(self,x):
        linear_pred=np.dot(x,self.weights)+self.bias
        y_pred=sigmoid(linear_pred)
        class_pred=[0 if y<=0.5 else 1 for y in y_pred]
        return class_pred