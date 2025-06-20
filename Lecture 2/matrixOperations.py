import numpy as np

#MATRIX ADDITION
A = np.array([[2,3] , [3,4]])
B = np.array([[3,4] , [5,5]])
# Sum = np.add(A,B)                         #BOTH THE SUM ARE SAME
# sum = A+B                                 #THE DIMENSIONS OF MATRIX TO BE ADDED SHOULD BE SAME
# print(Sum)
# print(sum)

#MATRIX SUBTRACTION
# A = np.array([[2,3] , [3,4]])
# B = np.array([[3,4] , [5,5]])
#DIFFERENCE = np.subtract(A,B)                         #BOTH THE DIFFERENCES ARE SAME
# differnce = A-B                                      # SAME CONDITION AS THE SUM OF MATRIX
# print(DIFFERENCE)
# print(differnce)

# MULTIPLYING A SCALAR WITH A MATRIX
# x = 5
# y = np.random.randint(10 , size=(3,3))
# print(y)
# product = np.multiply(x,y)
# print(product)

#MULTIPLYIG TWO MATRICES
x = np.random.randint(10 , size=(3,3))
y = np.random.randint(10 , size =(3,3))

#dot product
# dot_product = np.dot(x,y)
# print(dot_product)

# element wise multiplication
result = np.multiply(x,y)
print(result)