import numpy as np 
from tensorflow import keras 
from tensorflow.keras import layers 
np.random.seed(0) 
X = np.random.rand(100,1) 
y = 2*X.squeeze() + 1 + np.random.randn(100)*0.1 
model = keras.Sequential([ 
layers.Dense(10, activation='relu', input_shape=(1,)),layers.Dense(1) 
]) 
model.compile(optimizer='adam', loss='mse') 
model.fit(X, y, epochs=100, batch_size=32, verbose=0) 
pred = model.predict(X) 
for i in range(5): 
print("Predicted:", pred[i][0], "\tTrue:", y[i])