import pandas as pd

# leer database
dataset = pd.read_csv('heart.csv')


# leer  10 primeros registros
# print(dataset.head(10))

# print(dataset.info())

# leer 5 ultimos registros
# print(dataset.tail(5))

# agrupar datos por un campo
#group = dataset.groupby('species')

# ver longitud de cada grupo
# print(group.size())

# print(dataset.describe())


print('Sex {}'.format(dataset['Sex'].unique()))
print('ChestPainType {}'.format(dataset['ChestPainType'].unique()))
print('RestingECG {}'.format(dataset['RestingECG'].unique()))
print('ExerciseAngina {}'.format(dataset['ExerciseAngina'].unique()))
print('ST_Slope {}'.format(dataset['ST_Slope'].unique()))


# tranformar datos object  a numericos

datasetNumeric = {
    "sexNumeric" : [],
    "chestPainTypeNumeric" :[],
    "restingECGNumeric":[],
    "exerciseAnginaNumeric":[],
    "stSlopeNumeric": []
}

rowSize = len(dataset['HeartDisease'])


for index in range(rowSize):

    # sex
    sexCurrent = dataset['Sex'][index]

    if sexCurrent =='M':
        datasetNumeric['sexNumeric'].append(0)
    else : datasetNumeric['sexNumeric'].append(1)


    # chestPainTypeCurrent
    chestPainTypeCurrent = dataset['ChestPainType'][index]

    if  chestPainTypeCurrent == "ATA" :
         datasetNumeric['chestPainTypeNumeric'].append(0)

    elif  chestPainTypeCurrent == "NAP" :
        datasetNumeric['chestPainTypeNumeric'].append(1)

    elif  chestPainTypeCurrent == "ASY" :
        datasetNumeric['chestPainTypeNumeric'].append(2)
    
    elif  chestPainTypeCurrent == "TA" :
        datasetNumeric['chestPainTypeNumeric'].append(3)


    # RestingECG
    restingECGCurrent = dataset['RestingECG'][index]

    if restingECGCurrent =="Normal":
        datasetNumeric['restingECGNumeric'].append(0)

    elif restingECGCurrent =="ST":
         datasetNumeric['restingECGNumeric'].append(1)

    elif restingECGCurrent =="LVH":
         datasetNumeric['restingECGNumeric'].append(2)


    # ExerciseAngina
    exerciseAnginaCurrent =  dataset['ExerciseAngina'][index]

    if exerciseAnginaCurrent=="N":
        datasetNumeric['exerciseAnginaNumeric'].append(0)
    else : datasetNumeric['exerciseAnginaNumeric'].append(1)


    # ST_Slope
    stSlopeCurrent =dataset['ST_Slope'][index]

    if  stSlopeCurrent =="Up":
         datasetNumeric['stSlopeNumeric'].append(0)

    elif  stSlopeCurrent =="Flat":
        datasetNumeric['stSlopeNumeric'].append(1)

    else :
        datasetNumeric['stSlopeNumeric'].append(2)


# Verify number
# print( len (datasetNumeric['sexNumeric']))
# print( len (datasetNumeric['chestPainTypeNumeric']))
# print( len (datasetNumeric['restingECGNumeric']))
# print( len (datasetNumeric['exerciseAnginaNumeric']))
# print( len (datasetNumeric['stSlopeNumeric']))


# Depp copy of dataset in ordet to avoid changes in th real dataset
datasetHeart= dataset.copy(deep=True)

datasetHeart['Sex'] = datasetNumeric['sexNumeric']
datasetHeart['ChestPainType'] = datasetNumeric['chestPainTypeNumeric']
datasetHeart['RestingECG'] = datasetNumeric['restingECGNumeric']
datasetHeart['ExerciseAngina'] = datasetNumeric['exerciseAnginaNumeric']
datasetHeart['ST_Slope'] = datasetNumeric['stSlopeNumeric']


# print(dataset['Sex'].head(5))
# print(datasetHeart['Sex'].head(5))


# print(dataset['ChestPainType'].head(5))
# print(datasetHeart['ChestPainType'].head(5))


# print(dataset['RestingECG'].head(5))
# print(datasetHeart['RestingECG'].head(5))



# print(dataset['ExerciseAngina'].head(5))
# print(datasetHeart['ExerciseAngina'].head(5))



# print(dataset['ST_Slope'].head(5))
# print(datasetHeart['ST_Slope'].head(5))


# Save the cleaning dataset >>> numeric dataset
datasetHeart.to_csv('cleaningDataSet.csv') 


# Potling Dataset
print("****************** PLOTING DATASET ********")
import matplotlib.pyplot as plt 


# datasetHeart.plot(kind="box", subplots=True, layout=(4,4), sharex=False)
# plt.show()


# datasetHeart.hist()
# plt.show()

# Perfoming classification Task
print("****************Perfoming classification Task*****************")

# Eliminar caracteresticas no utilies, son caracteristccas
# datasetHeart = datasetHeart.drop(['Cholesterol', 'Oldpeak', 'RestingECG'] , axis =1 )

# print(datasetHeart.info())


# Eliminar una columna
x = datasetHeart.drop('HeartDisease',axis=1)

print(x.info())

y= dataset['HeartDisease']  

# preprocessing  DATA  INTO STANDARD  SCALE
from sklearn.preprocessing  import StandardScaler

scaler = StandardScaler()
X =  scaler.fit_transform(x)


# SPLITING TH DATASET INTO  TRAINING  AND  TESTING

from  sklearn.model_selection  import  train_test_split


# test_size = 0.7 : determina el pocenttaje para training y  0.3 para testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.7)


# determinar el algoritmo para training
from sklearn.ensemble import RandomForestClassifier

# RandomForestClassifier() es un algoritmo de clasifacion
model = RandomForestClassifier()

# training algoritmo
model.fit(X_train, y_train)




# Prediction
# conocer la predicion de los datos que ya se conocen y los datos de evaluacion y entrenamiento
yPredictionOnTraining = model.predict(X_train)
yPredictionOnTesting =model.predict(X_test)


from sklearn import  metrics


print('TRAIN  ACCURACY  SCORE: {}'.format(metrics.accuracy_score(y_train, yPredictionOnTraining)))
print("TEST  ACCURACY  SCORE: {}".format(metrics.accuracy_score(y_test, yPredictionOnTesting)))
print("CONFUSION  ACCURACY  SCORE: {}".format(metrics.confusion_matrix(y_test, yPredictionOnTesting)))


# *****************************  other algorithm **************************

# determinar el algoritmo para training
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression


# RandomForestClassifier() es un algoritmo de clasifacion
model = LogisticRegression()

# training algoritmo
model.fit(X_train, y_train)

# Prediction
# conocer la predicion de los datos que ya se conocen y los datos de evaluacion y entrenamiento
yPredictionOnTraining = model.predict(X_train)
yPredictionOnTesting =model.predict(X_test)


from sklearn import  metrics


print('TRAIN  ACCURACY  SCORE: {}'.format(metrics.accuracy_score(y_train, yPredictionOnTraining)))
print("TEST  ACCURACY  SCORE: {}".format(metrics.accuracy_score(y_test, yPredictionOnTesting)))
print("CONFUSION  ACCURACY  SCORE: {}".format(metrics.confusion_matrix(y_test, yPredictionOnTesting)))



# Lista de modelos de aprendizaje

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.svm import  SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import  metrics


from sklearn.neighbors import  KNeighborsClassifier
models = [ RandomForestClassifier(), KNeighborsClassifier(), 
           SVC(), LogisticRegression(),GaussianNB(), DecisionTreeClassifier(),
          MLPClassifier( hidden_layer_sizes = (8, 10,10,10), alpha = 0.0003, activation = 'relu', solver= 'adam', max_iter= 500) 
        ]

print("********************** MODELS *******************\n")
accuaryList= []
for model in models :
    # training algoritmo
    model.fit(X_train, y_train)

    # Prediction
    # conocer la predicion de los datos que ya se conocen y los datos de evaluacion y entrenamiento
    yPredictionOnTraining = model.predict(X_train)
    yPredictionOnTesting =model.predict(X_test)
    from sklearn import  metrics
    print('TRAIN  ACCURACY  SCORE: {}'.format(metrics.accuracy_score(y_train, yPredictionOnTraining)))
    print("TEST  ACCURACY  SCORE: {}".format(metrics.accuracy_score(y_test, yPredictionOnTesting)))
    print("CONFUSION  ACCURACY  SCORE: {}".format(metrics.confusion_matrix(y_test, yPredictionOnTesting)))

    accuaryList.append(metrics.accuracy_score(y_test, yPredictionOnTesting))


print(accuaryList)

# graficar resultados
datasetCompare = pd.DataFrame( {

    'Model' : ['RandomForestClassifier', 'KNeighborsClassifier', 'SVC', 'LogisticRegression', 'GaussianNB', 'DecisionTreeClassifier', 'MLPClassifier'],
    'Accuracy': accuaryList
})

# imprimir valores y ordenar
print(datasetCompare.sort_values(['Accuracy']))

