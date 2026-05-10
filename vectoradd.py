import tensorflow as tf 
a = tf.constant([1, 2, 3]) 
b = tf.constant([4, 5, 6]) 
c = tf.add(a, b) 
print("Vector A:", a.numpy()) 
print("Vector B:", b.numpy()) 
print("A + B =", c.numpy()) 