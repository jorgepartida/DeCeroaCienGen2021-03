
import pandas as pd
import numpy as np
from scipy import stats

x = np.array([12, 34, 67, 75, 91, 91, 132, 198])

print(np.mean(x))

print(np.median(x))

print(stats.mode(x))

print(x.max() - x.min())

print(np.var(x))

print(np.std(x))
'''
import numpy as np


x = np.array([3,7,9])
y = np.array([2,3,8])

z = np.array([5,7])
w = np.array([7,-5])

punto = np.dot(x,y)
print (punto)

punto = np.dot(z,w)
print (punto)

norma = np.linalg.norm(x)
print(norma)

norma = np.linalg.norm(y)
print(norma)

x = np.array([[2,7,3,5],
              [4,5,2,12],
             [1,2,1,-2]])

y = np.array([[1,-2],
             [4,7],
             [1,1],
             [8,9]])

z = x@y

print(z)

x = np.array([[-10,5,9],
              [7,12,-1],
              [8,3,1]])

print(np.linalg.det(x))
print(np.linalg.inv(x))


import pandas as pd
import numpy as np
from scipy import stats

x = np.array([9,3,8,8,9,8,9,18,3,7,9,1,0])

print(np.mean(x))

print(np.median(x))

print(stats.mode(x))

print(x.max() - x.min())

print(np.var(x))

print(np.std(x))



'''


