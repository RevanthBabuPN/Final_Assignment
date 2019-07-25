#Average_accuracy = 83%

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Dataset/Adelaide.csv')
df1=df.drop(['Date', 'Location','Evaporation', 'Sunshine','WindGustDir','WindDir9am','WindDir3pm','Cloud9am','Cloud3pm'],axis=1)

for index,item in df1.RainTomorrow.iteritems():
    if item=='No':
        df1.RainTomorrow.loc[index] = 0
    else:
        df1.RainTomorrow.loc[index] = 1

for index,item in df1.RainToday.iteritems():
    if item=='No':
        df1.RainToday.loc[index] = 0
    else:
        df1.RainToday.loc[index] = 1

for index,item in df1.WindSpeed3pm.iteritems():
    if np.isnan(item):
        df1.WindSpeed3pm.loc[index] = np.mean(df1.WindSpeed3pm)
for index,item in df1.WindSpeed9am.iteritems():
    if np.isnan(item):
        df1.WindSpeed9am.loc[index] = np.mean(df1.WindSpeed9am)
for index,item in df1.Humidity9am.iteritems():
    if np.isnan(item):
        df1.Humidity9am.loc[index] = np.mean(df1.Humidity9am)
for index,item in df1.Humidity3pm.iteritems():
    if np.isnan(item):
        df1.Humidity3pm.loc[index] = np.median(df1.Humidity3pm)
for index,item in df1.Pressure9am.iteritems():
    if np.isnan(item):
        df1.Pressure9am.loc[index] = np.median(df1.Pressure9am)
for index,item in df1.Pressure3pm.iteritems():
    if np.isnan(item):
        df1.Pressure3pm.loc[index] = np.median(df1.Pressure3pm)
for index,item in df1.Temp9am.iteritems():
    if np.isnan(item):
        df1.Temp9am.loc[index] = np.median(df1.Temp9am)
for index,item in df1.Temp3pm.iteritems():
    if np.isnan(item):
        df1.Temp3pm.loc[index] = np.median(df1.Temp3pm)

df1.dropna(inplace=True)

X = df1.iloc[:,0:14]
Y = df1.iloc[:,14]
sc = StandardScaler()
X = sc.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

model = keras.Sequential ([
    keras.layers.Dense(4, input_dim=14,activation='relu'),
    keras.layers.Dense(4, activation = 'relu'),
    keras.layers.Dense(1, activation = 'sigmoid')
])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
             metrics = ['accuracy'])
model.fit(X_train, Y_train, epochs = 5)
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(test_acc)

