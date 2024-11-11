# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required liberaries.
2.Load the dataset.
3.Define the X and Y array.
4.Define a cost function , cost and gradient .
5.Define a function to plot the decision boundary.
6.Define a function to predict the Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: ARAVIND G
RegisterNumber:  212223240011
*/
```
```
import numpy as np import matplotlib.pyplot as plt from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',') X=data[:,[0,1]] y=data[:,2]

X[:5]

y[:5]

plt.figure() plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted") plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted") plt.xlabel("Exam 1 score") plt.ylabel("Exam 2 score") plt.legend() plt.show()

def sigmoid(z): return 1/(1+np.exp(-z))

plt.plot() X_plot=np.linspace(-10,10,100) plt.plot(X_plot,sigmoid(X_plot)) plt.show()

def costFunction (theta,X,y): h=sigmoid(np.dot(X,theta)) J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0] grad=np.dot(X.T,h-y)/X.shape[0] return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X)) theta=np.array([0,0,0]) J,grad=costFunction(theta,X_train,y) print(J) print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X)) theta=np.array([-24,0.2,0.2]) J,grad=costFunction(theta,X_train,y) print(J) print(grad)

def cost (theta,X,y): h=sigmoid(np.dot(X,theta)) J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0] return J

def gradient (theta,X,y): h=sigmoid(np.dot(X,theta)) grad=np.dot(X.T,h-y)/X.shape[0] return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X)) theta=np.array([0,0,0]) res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient) print(res.fun) print(res.x)

def plotDecisionBoundary(theta,X,y): x_min,x_max=X[:,0].min()-1,X[:,0].max()+1 y_min,y_max=X[:,1].min()-1,X[:,1].max()+1 xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1)) X_plot=np.c_[xx.ravel(),yy.ravel()] X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot)) y_plot=np.dot(X_plot,theta).reshape(xx.shape)

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.contour(xx,yy,y_plot,levels=[0])
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()
plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x)) print(prob)

def predict(theta,X): X_train =np.hstack((np.ones((X.shape[0],1)),X)) prob=sigmoid(np.dot(X_train,theta)) return (prob>=0.5).astype(int) np.mean(predict(res.x,X)==y)
```


## Output:
Array Value of x

![Screenshot 2024-11-11 230923](https://github.com/user-attachments/assets/3982f3a3-9714-44e5-85c5-3dcc6e3ca976)

Array Value of y

![Screenshot 2024-11-11 231021](https://github.com/user-attachments/assets/89c71bd4-e5ae-4d3c-b88b-b1d77a11523d)

Exam 1 - score graph

![Screenshot 2024-11-11 231107](https://github.com/user-attachments/assets/a53ad351-a0a5-4905-ad31-3976f21cd17d)

Sigmoid function graph

![Screenshot 2024-11-11 231150](https://github.com/user-attachments/assets/087c29c6-a8cf-487a-bc90-aaf60171101a)

X_train_grad value

![Screenshot 2024-11-11 231226](https://github.com/user-attachments/assets/811cf487-c1c2-4a8f-ada7-d11ca456135e)

Y_train_grad value

![Screenshot 2024-11-11 231306](https://github.com/user-attachments/assets/fa8ca715-e26c-43e3-8212-958a4f5495de)

Print res.x

![Screenshot 2024-11-11 231416](https://github.com/user-attachments/assets/61bc3bcf-d840-4c58-b9c4-fcd4d9ba8b37)

Decision boundary - graph for exam score

![Screenshot 2024-11-11 231529](https://github.com/user-attachments/assets/a0067f18-4595-4a31-a5e2-84511abf210c)

Proability value

![Screenshot 2024-11-11 231604](https://github.com/user-attachments/assets/12377efc-df6b-4727-9448-de63a917a057)

Prediction value of mean

![Screenshot 2024-11-11 231639](https://github.com/user-attachments/assets/c6ce5673-ea98-495b-9b5d-21acade7b449)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

