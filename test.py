import pandas as pd

# leer database
dataset = pd.read_csv('IRIS.csv')

# leer  10 primeros registros
# print(dataset.head(10))

# leer 5 ultimos registros
# print(dataset.tail(5))

# agrupar datos por un campo
group = dataset.groupby('species')

# ver longitud de cada grupo
# print(group.size())

# print(dataset.describe())

print('********************* MATPLOTLIB ***********')
import matplotlib.pyplot as plt

# obtener datos de un columna
clasificacion = dataset['species']

# filtrar datos 
setosa=dataset[clasificacion=='Iris-setosa']
virginica=dataset[clasificacion=='Iris-virginica']
versicolor=dataset[clasificacion=='Iris-versicolor']

# print(setosa.head(5))
# print(virginica.head(5))
# print(versicolor.head(5))


# Grarficar

fig,ax=plt.subplots()
fig.set_size_inches(5,4)

#scatter diagrama de dispersion. relation between two variables 
#sepal_length  sepal_width  petal_length  petal_width
ax.scatter(setosa['petal_length'],setosa['petal_width'],facecolor='blue')
ax.scatter(virginica['petal_length'],virginica['petal_width'],facecolor='green')
ax.scatter(versicolor['petal_length'],versicolor['petal_width'],facecolor='pink')


# asignar etiquetas
ax.set_xlabel('petal length [cm]')
ax.set_ylabel('petal width [cm]')

ax.grid()
# asignar titulo
ax.set_title('IRIS PETALS')

# mostrar grafico
# plt.show()


print('******************* PERFORMING CLASSIFICATION  *********************************')

import sklearn
import numpy as np

# eliminar la colummna especie del dataset, axis significa columna
X=dataset.drop(['species'],axis=1)

# print(X.head(5))


# convertir  pandas.core.frame.DataFrame a un array numpy
X=X.to_numpy()

# dimension filas x columnas
# print(X.shape)

# seleccionar un rango de dato del arreglo
# :  = todas las filas
# (2, 3) = desde la columnas 2 hasta la columna 3
X=X[:,(2,3)]



# Para determinar la salidas, creara una lista de numeros que
#  representaran al los nombres de las flores
#  para que se haga mas facil el entranamiento

target=[]

for i in range(len(dataset['species'])):
    
    # 0 correspondera a Iris-setosa
    if dataset['species'][i]=='Iris-setosa':
        target.append(0)

    # 1 correspondera a Iris-versicolor
    elif dataset['species'][i]=='Iris-versicolor':
        target.append(1)

    # 2 correspondera a Iris-virginica
    else:
        target.append(2)


# convetir lista en un array numpy
y=np.array(target)
# print(y.shape)
# print(type(y)) 

print('////////////////////////////////////////////////////////////////////////')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Splitting into train and test
# datos de entrada y salida : test_size= % que se asigna al entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.75)

# crear modelo, LogisticRegression() es el algoritmo de Machine Learning
model=LogisticRegression()


print('**************  ENTREANAMIENTO *************')

# entrenar el modelo, le paso la entradas
model.fit(X_train,y_train)


print('--------------------------TEST PREDICTIONS-------------------------')
y_predict=model.predict(X_test)

# ver las 30 primeras predicciones
print(y_predict[:30])

# ver la 30 test para comparar con las predicciones
print(y_test[:30])

print('--------------------------------EVALUATIONS---------------------------')

from sklearn import metrics

print("Precision, Recall, Confusion matrix, in training")


# Precision Recall scores
print(metrics.classification_report(y_test,y_predict, digits=3))

print("CONFUSION MATRIX IN TESTING DATA")
print(metrics.confusion_matrix(y_test, y_predict))