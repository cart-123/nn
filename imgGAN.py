import tensorflow as tf,numpy as np,matplotlib.pyplot as p 
from tensorflow import keras 
(x,_),_=keras.datasets.mnist.load_data() 
x=((x-127.5)/127.5).reshape(-1,28,28,1) 
g=keras.Sequential([ 
keras.layers.Dense(128,input_dim=100), 
keras.layers.LeakyReLU(0.2), 
keras.layers.Dense(784,activation='tanh'), 
keras.layers.Reshape((28,28,1)) 
]) 
d=keras.Sequential([ 
keras.layers.Flatten(input_shape=(28,28,1)), 
keras.layers.Dense(128), 
keras.layers.LeakyReLU(0.2), 
keras.layers.Dense(1,activation='sigmoid') 
]) 
d.compile('adam','binary_crossentropy') 
d.trainable=0 
gan=keras.Sequential([g,d]) 
gan.compile('adam','binary_crossentropy') 
for i in range(10): 
r=x[np.random.randint(0,x.shape[0],32)] 
z=np.random.normal(0,1,(32,100)) 
f=g.predict(z,0) 
    dl=d.train_on_batch(r,np.ones((32,1))) 
    d.train_on_batch(f,np.zeros((32,1))) 
    gl=gan.train_on_batch(z,np.ones((32,1))) 
    print(f"Epoch {i+1}, D Loss: {dl}, G Loss: {gl}") 
 
img=g.predict(np.random.normal(0,1,(1,100))) 
p.imshow(img.reshape(28,28),cmap='gray') 
p.show()