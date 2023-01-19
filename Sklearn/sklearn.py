import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import operator
from  sklearn import preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split

def sortedList(x,y):
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x,y), key=sort_axis)
    x_sorted, y_sorted = zip(*sorted_zip)
    return x_sorted,y_sorted

def Average(lst):
    return sum(lst) / len(lst)

def regLineal(x,y,max_iter,learning_rate, eta0,scaling):
    regr = SGDRegressor(learning_rate = learning_rate, eta0 = eta0, max_iter= max_iter)

    #tipo de escalado
    if(scaling==0): 
        x = x
    elif(scaling==1): 
        x = preprocessing.StandardScaler().fit_transform(x)
    elif(scaling==2):
        x = preprocessing.RobustScaler().fit_transform(x)
    
    #entrenamos y predecimos con el modelo de acuerdo al tipo de escalado
    regr.fit(x,y.ravel())
    y_pred = regr.predict(x)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mse, r2, y_pred, regr

def regPolinomial(x,y,grado,max_iter,learning_rate, eta0, scaling):
    regr = SGDRegressor(learning_rate = learning_rate, eta0 = eta0, max_iter= max_iter)
    polynomial_features= PolynomialFeatures(degree=grado)
    x_poly = polynomial_features.fit_transform(x)


    #tipo de escalado
    if(scaling==0):
        x_scaled = x_poly 
    elif(scaling==1): 
        x_scaled = preprocessing.StandardScaler().fit_transform(x_poly)
    elif(scaling==2):
        x_scaled = preprocessing.RobustScaler().fit_transform(x_poly)
    
    #entrenamos y predecimos con el modelo de acuerdo al tipo de escalado
    regr.fit(x_scaled, y.ravel())
    y_poly_pred = regr.predict(x_scaled)
    mse = mean_squared_error(y, y_poly_pred)
    r2 = r2_score(y, y_poly_pred)
    return mse, r2, y_poly_pred, regr


# llamamos al excel
file_name='cal_housing.csv'
data =pd.read_csv(file_name)
#print(data.describe())
#print(data)

x_plot = data.drop(['medianHouseValue'],axis=1).values
y_plot = data['medianHouseValue'].values

x_trainF,x_testF,y_trainF,y_testF = train_test_split(x_plot,y_plot, test_size=0.2,random_state=0)

#guardamos datos de X para graficar la dispersion de los datos
xscatter0 = data['longitude'].values
xscatter1 = data['latitude'].values
xscatter2 = data['housingMedianAge'].values
xscatter3 = data['totalRooms'].values
xscatter4 = data['totalBedrooms'].values
xscatter5 = data['population'].values
xscatter6 = data['households'].values
xscatter7 = data['medianIncome'].values

fig = plt.figure(figsize=(15, 10))
ax1 = fig.add_subplot(2, 4, 1)
ax1.set_title("longitude")

ax2 = fig.add_subplot(2, 4, 2)
ax2.set_title("latitude")

ax3 = fig.add_subplot(2, 4, 3)
ax3.set_title("housingMedianAge")

ax4 = fig.add_subplot(2, 4, 4)
ax4.set_title("totalRooms")

ax5 = fig.add_subplot(2, 4, 5)
ax5.set_title("totalBedrooms")

ax6 = fig.add_subplot(2, 4, 6)
ax6.set_title("population")

ax7 = fig.add_subplot(2, 4, 7)
ax7.set_title("households")

ax8 = fig.add_subplot(2, 4, 8)
ax8.set_title("medianIncome")

ax1.scatter(xscatter0,y_plot, color='r',s=0.75)
ax2.scatter(xscatter1, y_plot, color='b',s=0.75)
ax3.scatter(xscatter2, y_plot, color='g',s=0.75)
ax4.scatter(xscatter3, y_plot, color='y',s=0.75)
ax5.scatter(xscatter4, y_plot, color='r',s=0.75)
ax6.scatter(xscatter5, y_plot, color='b',s=0.75)
ax7.scatter(xscatter6, y_plot, color='g',s=0.75)
ax8.scatter(xscatter7, y_plot, color='y',s=0.75)

#listas para guardar mse y r2 de cada regresion
MSEregLineal=[]
MSEregPoly2=[]
MSEregPoly3=[]
MSEregPoly1Estandar=[]
MSEregPoly2Estandar=[]
MSEregPoly3Estandar=[]
MSEregPoly1Robusto=[]
MSEregPoly2Robusto=[]
MSEregPoly3Robusto=[]

R2regLineal=[]
R2regPoly2=[]
R2regPoly3=[]
R2regPoly1Estandar=[]
R2regPoly2Estandar=[]
R2regPoly3Estandar=[]
R2regPoly1Robusto=[]
R2regPoly2Robusto=[]
R2regPoly3Robusto=[]

#lista para guadar el mejor valor de Y predicha
maxY_pred = 0
max_r2 = 0

columnasRegresiones= ['Regresión lineal',
                        'Regresión lineal escalado estandar',
                        'Regresión lineal escalado robusto',
                        'Regresión polinomial grado 2',
                        'Regresión polinomial grado 2 escalado estándar',
                        'Regresión polinomial grado 2 escalado robusto',
                        'Regresión polinomial grado 3',
                        'Regresión polinomial grado 3 escalado estándar',
                        'Regresión polinomial grado 3 escalado robusto']

k=10 #numero de pliegues previamente creados
for pliegue in range(k):
        print("Pliegue "+str(pliegue+1))
        file_namex='x_train_v'+str(pliegue+1)+'.csv'
        file_namey='y_train_v'+str(pliegue+1)+'.csv'
        datax = pd.read_csv(file_namex)
        datay = pd.read_csv(file_namey)
        #print(data.describe())
        #print(data)

        x=datax.values
        y=datay.values

        print("x shape init:", x.shape)
        print("y shape init:", y.shape)

        mse_list = []   
        r2_list = []

        #Modelo de regresión lineal
        mse, r2, y_pred, regr = regLineal(x,y,1000000,'constant',0.0001, 0)
        #print ('\n Regresión lineal: \nmse: {} r2: {}'.format(mse, r2), '\n')

        mse_list.append(mse)
        r2_list.append(r2)
        MSEregLineal.append(mse)
        R2regLineal.append(r2)

        #Modelo de regresión lineal escalado estandar
        mse, r2, y_pred, regr = regLineal(x,y,100000,'constant',0.0001, 1)
        #print ('\n Regresión lineal escalado estandar: \nmse: {} r2: {}'.format(mse, r2), '\n')
        mse_list.append(mse)
        r2_list.append(r2)
        MSEregPoly1Estandar.append(mse)
        R2regPoly1Estandar.append(r2)

        #Modelo de regresión lineal escalado robusto
        mse, r2, y_pred, regr = regLineal(x,y,100000,'constant',0.0001, 2)
        #print ('\n Regresión lineal escalado robusto:\nmse: {} r2: {}'.format(mse, r2), '\n')
        mse_list.append(mse)
        r2_list.append(r2)

        MSEregPoly1Robusto.append(mse)
        R2regPoly1Robusto.append(r2)


         # ~ #Modelo de regresión polinomial grado 2 
        mse, r2, y_poly_pred, regr = regPolinomial(x,y,2,100000,'constant',0.0001,0)
        #print ('\nRegresión polinomial grado 2: \nmse: {} r2: {}'.format(mse, r2),'\n')

        mse_list.append(mse)
        r2_list.append(r2)
        MSEregPoly2.append(mse)
        R2regPoly2.append(r2)

         # ~ #Modelo de regresión polinomial grado 2 estandar
        mse, r2, y_poly_pred, regr = regPolinomial(x,y,2,100000,'constant',0.0001,1)
        #print ('\nRegresión polinomial grado 2 escalado estandar: \nmse: {} r2: {}'.format(mse, r2),'\n')

        mse_list.append(mse)
        r2_list.append(r2),
        MSEregPoly2Estandar.append(mse)
        R2regPoly2Estandar.append(r2)
        
        #se anade el if para guardar el maximo r2 en esta regresion debido a que fue la que mejor puntaje obtuvo
        if r2 > max_r2: 
            max_r2=r2
            maxY_pred = (x,y,y_poly_pred,regr)

        # ~ #Modelo de regresión polinomial grado 2 escalado robusto
        
        mse, r2, y_poly_pred, regr = regPolinomial(x,y,2,100000,'constant',0.0001,2)
        #print ('Regresión polinomial grado 2 escalado robusto\nmse: {} r2: {}'.format(mse, r2))
        mse_list.append(mse)
        r2_list.append(r2)
        MSEregPoly2Robusto.append(mse)
        R2regPoly2Robusto.append(r2)

        # ~ ###########     EXPERIMENT 3

        # ~ # Modelo de regresión polinomial grado 3
        mse, r2, y_poly_pred, regr = regPolinomial(x,y,3,100000,'constant',0.0001,0)
        #print ('\nRegresión polinomial grado 3: \nmse: {} r2: {}'.format(mse, r2),'\n')

        mse_list.append(mse)
        r2_list.append(r2)
        MSEregPoly3.append(mse)
        R2regPoly3.append(r2)


        # ~ # Modelo de regresión polinomial grado 3 estandar
        mse, r2, y_poly_pred, regr = regPolinomial(x,y,3,100000,'constant',0.0001,1)
        #print ('\nRegresión polinomial grado 3 escalado estándar\nmse: {} r2: {}'.format(mse, r2),'\n')

        mse_list.append(mse)
        r2_list.append(r2)
        MSEregPoly3Estandar.append(mse)
        R2regPoly3Estandar.append(r2)

        # ~ # Modelo de regresión polinomial grado 3 robusto
        mse, r2, y_poly_pred, regr = regPolinomial(x,y,3,100000,'constant',0.0001,2)
        #print ('Regresión polinomial grado 3 escalado robusto\nmse: {} r2: {}'.format(mse, r2))

        mse_list.append(mse)
        r2_list.append(r2)
        MSEregPoly3Robusto.append(mse)
        R2regPoly3Robusto.append(r2)

        # Creates pandas DataFrame.
        data = {'mse':mse_list,
                'r2':r2_list}

        print("Resultados del pliegue k="+str(pliegue+1))
        df = pd.DataFrame(data, index = columnasRegresiones)
        print(df,'\n')

#promedios de pliegues
mse_list = []   
r2_list = []

mse_list.append(Average(MSEregLineal))
mse_list.append(Average(MSEregPoly1Estandar))
mse_list.append(Average(MSEregPoly1Robusto))
mse_list.append(Average(MSEregPoly2))
mse_list.append(Average(MSEregPoly2Estandar))
mse_list.append(Average(MSEregPoly2Robusto))
mse_list.append(Average(MSEregPoly3))
mse_list.append(Average(MSEregPoly3Estandar))
mse_list.append(Average(MSEregPoly3Robusto))
  
r2_list.append(Average(R2regLineal))
r2_list.append(Average(R2regPoly1Estandar))
r2_list.append(Average(R2regPoly1Robusto))
r2_list.append(Average(R2regPoly2))
r2_list.append(Average(R2regPoly2Estandar))
r2_list.append(Average(R2regPoly2Robusto))
r2_list.append(Average(R2regPoly3))
r2_list.append(Average(R2regPoly3Estandar))
r2_list.append(Average(R2regPoly3Robusto))

data = {'mse':mse_list, 'r2':r2_list}

# Creates pandas DataFrame de los promedios de los pliegues.
df = pd.DataFrame(data, index = columnasRegresiones)
print("---------------------------------------------------------------------------------")
print("Promedios de k="+str(k)+" pliegues: ")
print(df,'\n')

#obtenemos regresion con el mejor valor
max_value = max(r2_list)
index = r2_list.index(max_value)

#print(index)
print("La regresion con el valor mas alto fue la ",columnasRegresiones[index], ", con valor de: ", round(max_value, 4))
# print("x: ", maxY_pred[0])
# print("Y: ", maxY_pred[1])
# print("maxY_pred: ", maxY_pred[2])

#PROBAMOS EL MODELO PARA EL 20% DEL DATASET COMO TEST
#x_trainF,x_testF,y_trainF,y_testF
polynomial_features= PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(x_testF)

x_testFF = preprocessing.StandardScaler().fit_transform(x_poly)
regr.fit(x_testFF, y_testF.ravel())
y_predF = regr.predict(x_testFF)
#print ("Y predecida final:", y_predF)
mseF = mean_squared_error(y_testF, y_predF)
r2F = r2_score(y_testF, y_predF)
print ('\nRegresión polinomial grado 2 escalado estandar para el conjunto de prueba: \nmse: {} r2: {}'.format(mseF, r2F),'\n')


datax = maxY_pred[0]
y_plot = maxY_pred[1]
y_pred = maxY_pred[2]
regr = maxY_pred[3]

#print("data: ", data.shape)

xscatter0 = datax[:, 0]
xscatter1 = datax[:, 1]
xscatter2 = datax[:, 2]
xscatter3 = datax[:, 3]
xscatter4 = datax[:, 4]
xscatter5 = datax[:, 5]
xscatter6 = datax[:, 6]
xscatter7 = datax[:, 7]

fig2 = plt.figure(figsize=(15, 10))
gx1 = fig2.add_subplot(2, 4, 1)
gx1.set_title("longitude")
gx2 = fig2.add_subplot(2, 4, 2)
gx2.set_title("latitude")
gx3 = fig2.add_subplot(2, 4, 3)
gx3.set_title("housingMedianAge")
gx4 = fig2.add_subplot(2, 4, 4)
gx4.set_title("totalRooms")
gx5 = fig2.add_subplot(2, 4, 5)
gx5.set_title("totalBedrooms")
gx6 = fig2.add_subplot(2, 4, 6)
gx6.set_title("population")
gx7 = fig2.add_subplot(2, 4, 7)
gx7.set_title("households")
gx8 = fig2.add_subplot(2, 4, 8)
gx8.set_title("medianIncome")


x1,x2 = sortedList(xscatter0,y_pred)
gx1.plot(x1, x2, color='orange',linewidth=1)
gx1.scatter(xscatter0,y_plot, color='red',s=0.75)
gx1.scatter(xscatter0,y_pred, color='purple',s=0.75, marker="x")

x1,x2 = sortedList(xscatter1,y_pred)
gx2.plot(x1, x2, color='orange',linewidth=1)
gx2.scatter(xscatter1,y_plot, color='blue',s=0.75)
gx2.scatter(xscatter1,y_pred, color='purple',s=0.75, marker="x")

x1,x2 = sortedList(xscatter2,y_pred)
gx3.plot(x1, x2, color='orange',linewidth=1)
gx3.scatter(xscatter2,y_plot, color='green',s=0.75)
gx3.scatter(xscatter2,y_pred, color='purple',s=0.75, marker="x")

x1,x2 = sortedList(xscatter3,y_pred)
gx4.plot(x1, x2, color='orange',linewidth=1)
gx4.scatter(xscatter3,y_plot, color='yellow',s=0.75)
gx4.scatter(xscatter3,y_pred, color='purple',s=0.75, marker="x")

x1,x2 = sortedList(xscatter4,y_pred)
gx5.plot(x1, x2, color='orange',linewidth=1)
gx5.scatter(xscatter4,y_plot, color='red',s=0.75)
gx5.scatter(xscatter4,y_pred, color='purple',s=0.75, marker="x")

x1,x2 = sortedList(xscatter5,y_pred)
gx6.plot(x1, x2, color='orange',linewidth=1)
gx6.scatter(xscatter5,y_plot, color='blue',s=0.75)
gx6.scatter(xscatter5,y_pred, color='purple',s=0.75, marker="x")

x1,x2 = sortedList(xscatter6,y_pred)
gx7.plot(x1, x2, color='orange',linewidth=1)
gx7.scatter(xscatter6,y_plot, color='green',s=0.75)
gx7.scatter(xscatter6,y_pred, color='purple',s=0.75, marker="x")

x1,x2 = sortedList(xscatter7,y_pred)
gx8.plot(x1, x2, color='orange',linewidth=1)
gx8.scatter(xscatter7,y_plot, color='yellow',s=0.75)
gx8.scatter(xscatter7,y_pred, color='purple',s=0.75, marker="x")

plt.show()