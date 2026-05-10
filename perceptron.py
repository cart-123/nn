import numpy as np 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
X = np.array([[0,0],[0,1],[1,0],[1,1]]) 
y = np.array([0,1,1,1]) 
model = Sequential([Dense(1, input_shape=(2,), activation='sigmoid')]) 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
model.fit(X, y, epochs=1000, verbose=0) 
loss, acc = model.evaluate(X, y) 
print("Loss:", loss) 
print("Accuracy:", acc) 
print("Predictions:", model.predict(X).flatten())