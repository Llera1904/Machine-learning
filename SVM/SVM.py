import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("heart.csv", sep=',', engine='python')
x = df.drop('target', axis=1).values # Corpus sin etiquetas 
y = df['target'].values # Etiquetas
print(df)

# Separa corpus en conjunto de entrenamiento y prueba
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=0) # 30% prueba 70% entrenamiento

positive = []
negative = []
sumPositive = 0
sumNegative = 0
for row in range(len(yTrain)): # Separa en negativos y positivos 
    if  yTrain[row] == 1:
        positive.append(xTrain[row])
    elif  yTrain[row] == 0:
        negative.append(xTrain[row])

cPositive = np.sum(positive, axis=0) / len(positive)
cNegative = np.sum(negative, axis=0) / len(negative)
twoC = np.add(cPositive, cNegative)
c = twoC / 2
cNorm = np.linalg.norm(c)
print("\ncPositive:\n", cPositive)
print("\ncNegative:\n", cNegative)
print("\n2C:\n", twoC)
print("\nC:\n ", c)
print("\nCnorm:\n", cNorm)
    
# Prueba
resultTest = np.dot(xTest, c) / cNorm
print("\nresultTest:\n", resultTest)

yPred = []
for sigma in resultTest:
    if sigma < cNorm:	
        yPred.append(0)
    elif sigma >= cNorm:
        yPred.append(1)
print("\nyPred:\n", yPred)

# Metricas
normalizedAccuracy = accuracy_score(yTest, yPred)
accuracy = accuracy_score(yTest, yPred, normalize=False)
print ("\nnormalizedAccuracy:", normalizedAccuracy * 100, "%")
print ("Accuracy:", accuracy)

targetNames = ['0', '1']
cm = confusion_matrix(yTest, yPred, labels=[0,1])
print(classification_report(yTest, yPred, target_names=targetNames))
print (cm)

disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=targetNames)
disp1.plot()
plt.show()



