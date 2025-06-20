import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# plt.quiver(0,0,4,5)
# plt.show()

# plt.quiver(0,0,4,5 , scale_units ='xy' , angles='xy' , scale=1 , color='r')
# plt.quiver(0,0,-3,-6   , scale_units ='xy' , angles='xy' , scale=1 , color='b')
# plt.xlim(-6,6)
# plt.ylim(-6,6)
# plt.show()


# VECTOR ADDITION

# vector_1 = np.asarray([0,0,1,2])
# vector_2 = np.asarray([0,0,2,1])
# sum = vector_1+vector_2
# print(sum)


# VECTOR SUBTRACTION

# vector_1 = np.asarray([0,0,1,2])
# vector_2 = np.asarray([0,0,2,1])
# difference = vector_1-vector_2
# print(difference)

#MULTIPLYING A VECTOR BY A SCALAR

vector_1 = np.asarray([0,0,1,2])
vector_2 = 3*vector_1
print(vector_2)