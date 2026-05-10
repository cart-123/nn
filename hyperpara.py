import numpy as np 
from tensorflow import keras 
from sklearn.model_selection import train_test_split 
 
X=np.random.rand(500,1) 
y=3*X+2 
 
x1,x2,y1,y2=train_test_split(X,y,test_size=0.2) 
 
for e in [10,50,100]: 
    m=keras.Sequential([ 
        keras.layers.Dense(10,activation='relu',input_shape=(1,)), 
        keras.layers.Dense(1) 
    ]) 
    m.compile('adam','mse') 
    m.fit(x1,y1,epochs=e,verbose=0) 
    print("Epochs:",e,"Loss:",m.evaluate(x2,y2,verbose=0)) 
 
 
 
 
 
 
 
 
 
 
