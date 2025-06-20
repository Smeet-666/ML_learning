import numpy as np

# matrix_1 = np.array([[2,3] , [4,5]])
# print(matrix_1)
# print(matrix_1.shape)

# CREATING A MATRIX WITH RANDOM VALUE
# random_matrix = np.random.rand(3,3)*100
# print(random_matrix)

# # CREATING RANDOM INTEGER MATRIX
# random_int_matrix = np.random.randint(100,size=(3,3))
# print(random_int_matrix)

# CREATING AN UNIT MATRIX
# unit = np.ones((2,3), dtype=int)
# print(unit)

# CREATING AN NULL MATRIX
# null = np.zeros((2,3), dtype=int)
# print(null)


# CREATING AN IDENTITY MATRIX
# identity = np.eye(2,3)
# print(identity)


# CREATING A TRANSPOSE OF THE MATRIX
matrix = np.random.randint(20 , size=(3,3))
print(matrix)
transpose = np.transpose(matrix)
print(transpose)