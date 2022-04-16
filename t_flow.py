import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Sequential

dataFrame = pd.read_excel("bisiklet_fiyatlari.xlsx")
# result = sbn.pairplot(dataFrame)
# plt.show()

#!!! veriyi test/train olarak ayirmak

# y = wx + b
# y => label (gidilmek istenen nokta/sonuç). (BU ÖRNEKTE FİYAT).
# x => feature (özellik).

y = dataFrame["Fiyat"].values # fiyat bilgileri bir numpy dizisine çevrildi
# print(y)

x = dataFrame[["BisikletOzellik1", "BisikletOzellik2"]].values # özellikler numpy dizisine çevrildi
# print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=15)
# print(x_train.shape)
# print(x_test.shape)

#!!! scaling (boyutunu değiştirmek, büyütmek veya küçültmek) nöronlara verilecek veri setinin boyutunu değiştirmek için kullanilir.

scaler = MinMaxScaler()
scaler.fit_transform(x_train)
scaler.fit_transform(x_test)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# verilerin hepsi scale edildi yani 0-1 arasina çekildi.
# print(x_train)

# modeli oluştumak

model = Sequential()
model.add(Dense(4,activation="relu")) # 4 nöronlu bir hidden layer
model.add(Dense(4,activation="relu")) # 4 nöronlu bir hidden layer
model.add(Dense(4,activation="relu")) # 4 nöronlu bir hidden layer

model.add(Dense(1)) # 1 nöronlu bir output layer

model.compile(optimizer="rmsprop", loss="mse")

# modeli eğitmek

model.fit(x_train, y_train, epochs=250)

loss = model.history.history["loss"] # [loss] yazilmasinin sebebi dict'ten cikarmak ve dizi haline getirmektir.
result2 = sbn.lineplot(x=range(len(loss)), y=loss)
# plt.show()

trainLoss = model.evaluate(x_train, y_train, verbose=0)
trainTest = model.evaluate(x_test, y_test, verbose=0)

# print(trainLoss, trainTest) # yakin olmalari gerekir.

testTahminleri = model.predict(x_test)
print(testTahminleri)













