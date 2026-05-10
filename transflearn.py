import tensorflow as tf, numpy as np 
from tensorflow.keras import layers,models 
from tensorflow.keras.applications import MobileNetV2 
(x,y),(xt,yt)=tf.keras.datasets.cifar10.load_data() 
i=np.where((y==3)|(y==5))[0] 
j=np.where((yt==3)|(yt==5))[0] 
x,y=x[i],(y[i]==5).astype(int) 
xt,yt=xt[j],(yt[j]==5).astype(int) 
x=np.array([tf.image.resize(a,(64,64)).numpy() for a in x])/255.0 
xt=np.array([tf.image.resize(a,(64,64)).numpy() for a in xt])/255.0 
b=MobileNetV2(input_shape=(64,64,3),include_top=0,weights='imagenet') 
b.trainable=0 
m=models.Sequential([b,layers.GlobalAveragePooling2D(),layers.Dense(1,activation='sigmoid')]) 
m.compile('adam','binary_crossentropy',metrics=['accuracy']) 
m.fit(x,y,epochs=3,validation_data=(xt,yt)) 
p=m.predict(xt[:1]) 
print("Prediction:","DOG" if p[0][0]>0.5 else "CAT")