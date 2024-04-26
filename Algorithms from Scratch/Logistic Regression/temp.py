class MyClass:
    def __init__(self, x):
        self.x = x-1

    def print_value(self):
        print(self.x)

# Creating an instance of MyClass
obj = MyClass(10)

# Calling the method print_value() on the instance
obj.print_value()  # Output: 10


import numpy as np

x=np.array([[1,2,3],[1,2,3],[1,2,3],[1,2,3]])
print(np.shape(x))

sample,features=np.shape(x)

w=np.zeros(features)
print(w)

