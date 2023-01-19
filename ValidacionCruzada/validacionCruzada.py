import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import sys
import pickle

class validationSet:
	def __init__(self, xTrain, yTrain, xTest, yTest):
		self.xTrain = xTrain
		self.yTrain = yTrain
		self.xTest = xTest
		self.yTest = yTest

class testSet:
	def __init__(self, xTest, yTest):
		self.xTest = xTest
		self.yTest = yTest

class trainSet:
	def __init__(self, xTrain, yTrain):
		self.xTrain = xTrain
		self.yTrain = yTrain

class dataSet:
	def __init__(self, validationSet, testSet):
		self.validationSet = validationSet
		self.testSet = testSet

def generateTrainTest(fileName, k):
    # ~ pd.options.display.max_colwidth = 200 # Define el acho de las columnas (ancho máximo por default 50 caracteres)		
	# Lee el corpus original del archivo de entrada y lo pasa a un DataFrame
	df = pd.read_csv(fileName, sep=',', engine='python')
	x = df.drop('RainTomorrow', axis=1).values # Corpus sin etiquetas 
	y = df['RainTomorrow'].values # Etiquetas
	
	# Separa corpus en conjunto de entrenamiento y prueba
	xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, shuffle=True)	# 20% prueba 80% entrenamineto
	
	# print("\nDataset\n", df)
	# print("\nCorpus\n", *x)
	# print("\nEtiquetas\n", *y)
	# print("\n Conjunto de prueba")
	# print("\n X_test\n", *xTest)
	# print("\n y_test\n", *yTest)
	# print("\nConjunto de entrenamiento = Conjunto de Validación")
	# print("\n X_train\n", *xTrain)
	# print("\n y_train\n", *yTrain)

    # Crea pliegues para la validación cruzada
	# print("\nVALIDACION CRUZADA k=2\n")

	validationSets = []
	kf = KFold(n_splits=k) # Numero de pliegues
	i = 0

	for trainIndex, testIndex in kf.split(xTrain):
		i += 1
		xTrainV, xTestV = xTrain[trainIndex], xTrain[testIndex]
		yTrainV, yTestV = yTrain[trainIndex], yTrain[testIndex]
		validationSets.append(validationSet(xTrainV, yTrainV, xTestV, yTestV)) # Agrega el pliegue creado a la lista

		# print("\nPLIEGUE", i ,"\n")
		# print("xTrainV", *xTrainV, "\nyTrainV", *yTrainV, "\n")
		# print("xTestV", *xTestV, "\nyTestV", *yTestV)

	# Almacena el conjunto de prueba Y entrenamiento
	myTestSet = testSet(xTest, yTest)
	# myTrainSet = trainSet(xTrain, yTrain)

    # Guarda el dataset con los pliegues del conjunto de validación y el conjunto de pruebas
	myDataSet = dataSet(validationSets, myTestSet) 

	return myDataSet
	
if __name__=='__main__':
	k = 10 # Numero de pliegues 
	myDataSet = generateTrainTest("./Practica2/weatherAUS.csv", k)

	# Guarda el conjunto de prueba en formato csv
	formatData = "%s"
	np.savetxt("data_test.csv", myDataSet.testSet.xTest, delimiter=",", fmt=formatData, header="Date,Location,MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday")
	np.savetxt("target_test.csv", myDataSet.testSet.yTest, delimiter=",", fmt=formatData, header="RainTomorrow")

	i = 1
	for valSet in myDataSet.validationSet:
		# Entrenamiento 
		np.savetxt("data_validation_train_" + str(i) + "_" + str(k) + ".csv", valSet.xTrain, delimiter=",", fmt=formatData, header="Date,Location,MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday", comments="")
		np.savetxt("target_validation_train_" + str(i) + "_" + str(k) + ".csv", valSet.yTrain, delimiter=",", fmt=formatData, header="RainTomorrow", comments="")
		# Prueba
		np.savetxt("data_test_" + str(i) + "_" + str(k) + ".csv", valSet.xTest, delimiter=",", fmt=formatData, header="Date,Location,MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday", comments="")
		np.savetxt("target_test_" + str(i) + "_" + str(k) + ".csv", valSet.yTest, delimiter=",", fmt=formatData, header="RainTomorrow", comments="")
		i += 1

    # Guarda el dataset en pickle
	datasetFile = open('dataset.pkl', 'wb')
	pickle.dump(myDataSet, datasetFile)
	datasetFile.close()
	
	datasetFile = open('dataset.pkl', 'rb')
	myDataSetPickle = pickle.load(datasetFile)
	# print (*myDataSetPickle.testSet.Xtest)

	
    
	
	