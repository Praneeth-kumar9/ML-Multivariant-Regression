import numpy as np
X1 = []
y1 = []
a= []
import csv
f = open('housing.csv')
csv_f  = csv.reader(f)
for Row in csv_f:
    X1.append([float(Row[0]),float(Row[1]),float(Row[2]),float(Row[3]),float(Row[4]),float(Row[5]),float(Row[6]),float(Row[7]),float(Row[8]),float(Row[9]),float(Row[10]),float(Row[11]),float(Row[12])])
    y1.append([float(Row[13])])
    a.append([float(Row[13])])
f.close()
X1= X1/np.amax(X1, axis=0)
c =np.amax(y1, axis=0)
y1= y1/np.amax(y1, axis=0)
X=np.array(X1)
y=np.array(y1)
class NeuralNetwork(object):
    def __init__(self):
        self.lrate = 0.01
        self.inputsize = 13
        self.outputsize = 1
        self.W1 = np.random.randn(self.inputsize, self.outputsize)
        self.b = 0
    def feedForward(self,X):
        self.z = np.dot(X,self.W1) + self.b
        output = self.sigmoid(self.z)
        return output
        
    def sigmoid(self, s, deriv=False):
        if (deriv == True):
            return s * (1 - s) * self.lrate
        return 1/(1 + np.exp(-s))
        
    def backward(self,X,y,output):
        self.output_error = output-y
        self.output_delta = self.output_error * self.sigmoid(output, deriv=True)
        self.W1 -=  np.dot(X.T,self.output_delta)  
        self.b -=  np.sum(self.output_delta) 
    
    def train(self, X, y):
        output = self.feedForward(X)
        self.backward(X, y, output)
        
NN = NeuralNetwork()



for i in range(10000): #trains the NN 10000 times
    if (i % 1000 == 0):
        loss = (np.sqrt(np.mean(np.square(y - NN.feedForward(X)))))
        print("Loss: " + str(loss))
    NN.train(X, y)
print("Accuracy:" + str(100 - (loss * 100)))
Z = NN.feedForward(X)
print(str("predicted") + "\t\t\t\t\t\t" + str("actual"))
for i in range(len(Z)):
    print(str(Z[i]*c) + "\t\t\t\t\t\t" + str(a[i]))