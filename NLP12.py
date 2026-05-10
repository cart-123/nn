import numpy as np 
 
s=["hi","bye","thanks"] 
a=["hello","goodbye","welcome"] 
Q=np.zeros((3,3)) 
x,g,e=.1,.9,.2 
 
def r(i,j): 
    return 1 if i==j else -1 
 
for _ in range(100): 
    i=np.random.randint(3) 
    j=np.random.randint(3) if np.random.rand()<e else np.argmax(Q[i]) 
    Q[i,j]+=x*(r(i,j)+g*np.max(Q[i])-Q[i,j]) 
print("Enter: hi / bye / thanks") 
u=input() 
if u in s: 
    i=s.index(u) 
    print("Bot:",a[np.argmax(Q[i])]) 
else: 
    print("Unknown input")