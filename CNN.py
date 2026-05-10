import tensorflow as tf, matplotlib.pyplot as plt 
from tensorflow.keras import datasets,layers,models 
(X_train,y_train),(X_test,y_test)=datasets.cifar10.load_data() 
X_train,X_test=X_train/255.0,X_test/255.0 
c=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'] 
plt.figure(figsize=(10,10)) 
for i in range(25): 
plt.subplot(5,5,i+1) 
plt.xticks([]); 
plt.yticks([]); 
plt.grid(False) 
plt.imshow(X_train[i]) 
plt.xlabel(c[y_train[i][0]]) 
plt.show() 
model=models.Sequential([ 
layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)), 
layers.MaxPooling2D(2,2), 
layers.Conv2D(64,(3,3),activation='relu'), 
layers.MaxPooling2D(2,2), 
layers.Conv2D(64,(3,3),activation='relu'), 
layers.Flatten(), 
layers.Dense(64,activation='relu'), 
layers.Dense(10) 
]) 
model.compile('adam', 
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
metrics=['accuracy']) 
h=model.fit(X_train,y_train,epochs=10,validation_data=(X_test,y_test)) 
plt.plot(h.history['accuracy'],label='train') 
plt.plot(h.history['val_accuracy'],label='val') 
plt.xlabel('Epoch'); 
plt.ylabel('Accuracy') 
plt.legend(); 
plt.show() 
print("Test accuracy:",model.evaluate(X_test,y_test,verbose=0)[1])