import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import sys
import pickle

class testSet:
	def __init__(self, xTest, yTest):
		self.xTest = xTest
		self.yTest = yTest

class trainSet:
	def __init__(self, xTrain, yTrain):
		self.xTrain = xTrain
		self.yTrain = yTrain

def generateTrainTest(fileName):
    # ~ pd.options.display.max_colwidth = 200 # Define el acho de las columnas (ancho máximo por default 50 caracteres)		
	# Lee el corpus original del archivo de entrada y lo pasa a un DataFrame
	df = pd.read_csv(fileName, sep=',', engine='python')
	x = df.drop('target', axis=1).values # Corpus sin etiquetas 
	y = df['target'].values # Etiquetas
	
	# Separa corpus en conjunto de entrenamiento y prueba
	xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.4, shuffle=False)	# 40% prueba 60% entrenamiento
	
	# print("\nDataset\n", df)
	# print("\nCorpus\n", *x)
	# print("\nEtiquetas\n", *y)
	# print("\n Conjunto de prueba")
	# print("\n X_test\n", *xTest)
	# print("\n y_test\n", *yTest)
	# print("\nConjunto de entrenamiento = Conjunto de Validación")
	# print("\n X_train\n", *xTrain)
	# print("\n y_train\n", *yTrain)
	
	# Almacena el conjunto de prueba Y entrenamiento
	myTestSet = testSet(xTest, yTest)
	myTrainSet = trainSet(xTrain, yTrain)

	return myTestSet, myTrainSet
	
if __name__=='__main__':
	myTestSet, myTrainSet = generateTrainTest("heart.csv")
	
	# Guarda el conjunto de prueba y entrenamiento en formato csv
	np.savetxt("x_p.csv", myTestSet.xTest, delimiter=",", fmt="%d", header="age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal")
	np.savetxt("y_p.csv", myTestSet.yTest, delimiter=",", fmt="%d", header="target")
	np.savetxt("x_t.csv", myTrainSet.xTrain, delimiter=",", fmt="%d", header="age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal")
	np.savetxt("y_t.csv", myTrainSet.yTrain, delimiter=",", fmt="%d", header="target")
    
	
	