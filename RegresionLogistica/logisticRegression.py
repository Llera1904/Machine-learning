import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
df = pd.read_csv('breast-cancer.csv', sep=',', engine='python')
X = df.drop(['diagnosis', 'id'], axis=1).values   
y = df['diagnosis'].values
# plt.scatter(X,y)
	
# Separa el corpus cargado en el DataFrame en el 90% para entrenamiento y el 10% para pruebas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=0)	
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test) # La clase predicha
print ('\n Clase real', y_test)
print ('\n Clase predicha', y_pred,'\n\n')

print('\nMatriz de confusión')
print(confusion_matrix(y_test, y_pred))
print('\nAccuracy')
print('Porcentaje de instancias predichas correctamente', accuracy_score(y_test, y_pred)) 
print('Cantidad de instancias predichas correctamente', accuracy_score(y_test, y_pred, normalize=False), '\n\n') 