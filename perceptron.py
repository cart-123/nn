import matplotlib.pyplot as plt 
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Flatten,Dense 
(X_train,y_train),(X_test,y_test)=mnist.load_data() 
X_train,X_test=X_train/255.0,X_test/255.0 
model=Sequential([ 
Flatten(input_shape=(28,28)), 
Dense(128,activation='relu'), 
Dense(10,activation='softmax') 
]) 
model.compile('adam','sparse_categorical_crossentropy',metrics=['accuracy']) 
h=model.fit(X_train,y_train,epochs=5,validation_split=0.2) 
loss,acc=model.evaluate(X_test,y_test) 
print("Accuracy:",acc) 
plt.plot(h.history['accuracy'],label='train') 
plt.plot(h.history['val_accuracy'],label='validation') 
plt.legend() 
plt.title("Model Accuracy") 
plt.show() 
