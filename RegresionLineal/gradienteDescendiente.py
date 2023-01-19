import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import sys

def F(w, X, y):
	return sum((w * x - y)**2 for x, y in zip(X, y)) / len(y)

def dF(w, X, y):
	return sum(2*(w * x - y) * x for x, y in zip(X, y)) / len(y)

def printLine(points, w, iteration, ax, lineColor=None, lineStyle='dotted'):
	listX= []
	listY = []
	for index, tuple in enumerate(points):
		x = tuple[0]
		y = x * w
		listX.append(x)
		listY.append(y)
	ax.text(x, y, iteration, horizontalalignment='right')
	ax.plot(listX, listY, color=lineColor, linestyle=lineStyle)

if __name__=='__main__':
	# X = [1, 2, 3, 4, 5, 6]
	# y = [1, 2.5, 2, 4, 4.5, 6.3]

	fileName = "./Practica3/dataset_ejercicio_I_regresion_lineal.csv"
	df = pd.read_csv(fileName, sep=',', engine='python')
	X = list(df['size'].values )						
	y = list(df['price'].values) 
	xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.1, random_state=0)
	print("xTrain: ", xTrain)
	print("yTrain: ", yTrain)
	print("xTest: ", xTest)
	print("yTest: ", yTest)

	X = xTrain
	y = yTrain
	# X = xTest
	# y = yTest

	listError = []
	listW = []	
	# iterations = int(sys.argv[1])
	
	fig = plt.figure(figsize=(15, 5))
	ax1 = fig.add_subplot(1, 2, 1)
	ax1.set_title("Linear regression")
	ax1.set(xlabel="size", ylabel="price")
	ax2 = fig.add_subplot(1, 2, 2)
	ax2.set_title("Loss function")
	ax2.set(xlabel="weight", ylabel="error")
	
	ax1.scatter(X, y)
	
	w = 0
	alpha = 0.000001
	iterations = 150
	for t in range(iterations):
		error = F(w, X, y)
		gradient = dF(w, X, y)
		# print ('gradient = {}'.format(gradient))
		ax2.scatter(w, error)
		ax2.text(w, error, t, horizontalalignment='right')
		listW.append(w)
		listError.append(error)
		
		w = w - alpha * gradient
		# print ('iteration {}: w = {}, F(w) = {}'.format(t, w, error))
		printLine(zip(X, y), w, t, ax1)
		
	printLine(zip(X, y), w, t, ax1, 'red', 'solid')
	ax2.plot(listW, listError, color = 'red', linestyle = 'solid')

	yPred = []
	for xResult in xTest:
		yResult = w * xResult
		yPred.append(yResult)
	print("Datos reales = ", yTest)
	print("Predicciones = ", yPred)

	errorTest = F(w, xTest, yTest)
	print("MSE de entrenamiento: ", error)
	print("MSE de prueba: ", errorTest)
	print("Peso final: ", w)

	fig2 = plt.figure(figsize=(5, 5))
	ax3 = fig2.add_subplot(1, 1, 1)
	ax3.set_title("Linear regression test set")
	ax3.set(xlabel="size", ylabel="price")
	ax3.scatter(xTest, yTest, color='orange')
	ax3.scatter(xTest, yPred, color='purple')
	printLine(zip(xTest, yPred), w, 0, ax3, 'yellow', 'solid')
	
	plt.show()
