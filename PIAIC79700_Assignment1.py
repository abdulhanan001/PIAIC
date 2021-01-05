# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[1]:


import numpy as np


# 2. Create a null vector of size 10 

# In[3]:


array = np.zeros(10)



# 3. Create a vector with values ranging from 10 to 49

# In[6]:


arr = np.arange(10,50)



# 4. Find the shape of previous array in question 3

# In[7]:


print(arr.shape)


# 5. Print the type of the previous array in question 3

# In[9]:


print(type(arr))


# 6. Print the numpy version and the configuration
# 

# In[13]:


print(np.__version__,np.show_config())


# 7. Print the dimension of the array in question 3
# 

# In[14]:


print(arr.ndim)


# 8. Create a boolean array with all the True values

# In[16]:


bool = np.ones((2, 2), dtype=bool)


# 9. Create a two dimensional array
# 
# 
# 

# In[36]:


two_d = np.ones((2,3))


# ### 10. Create a three dimensional array
# 
# 

# In[38]:


three_D = np.zeros((2,3,3))


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[40]:


arr = np.arange(1,10)
print(arr[::-1])


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[44]:


null = np.zeros(10)
null[5] = 1
print(null)


# 13. Create a 3x3 identity matrix

# In[47]:


ey = np.eye(3,3)



# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[54]:


arr = np.array([1, 2, 3, 4, 5],dtype=float)
print(arr.dtype)


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[71]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]])
arr2 = np.array([[0., 4., 1.],

           [7., 2., 12.]])

mul = np.multiply(arr1,arr2)
dot = np.dot(arr1,arr2.T)


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[75]:


arr1 = np.array([[1., 2., 3.],[4., 5., 6.]])
arr2 = np.array([[0., 4., 1.],[7., 2., 12.]])
print(arr1 == arr2)


# 17. Extract all odd numbers from arr with values(0-9)

# In[86]:


arr = np.arange(10)
print(arr[1:10:2])


# 18. Replace all odd numbers to -1 from previous array

# In[88]:


arr[1:10:2] = -1


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[96]:


arr = np.arange(10)
arr[[5,6,7,8]] = 12


# 20. Create a 2d array with 1 on the border and 0 inside

# In[110]:


ar = np.ones((4,4))
ar[1:-1,1:-1] = 0


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[120]:


arr2d = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
arr2d[1][1] = 12


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[151]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[[0][0]][0] =64


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[173]:


tD = np.arange(10).reshape(2,5)
print(tD[0][1])


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[174]:


tD1 = np.arange(10).reshape(2,5)
print(tD1[1][2])


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[177]:


tD2 = np.arange(10).reshape(2,5)
print(tD2[:,3])


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[186]:


rand = np.random.randn(10,10)
print(rand.shape)
print(np.max(rand))
print(np.min(rand))


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[190]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
print(np.intersect1d(a,b)) 


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[192]:


a = np.intersect1d(a,b)


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[200]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
print(names == 'Will')
print(data[~(names == 'Will')])


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[204]:


mask = (names == 'Will') & (names == 'Joe')
print(data[~mask])


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[220]:


arr = np.arange(1,16).reshape(5,3).astype(np.float64)


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[225]:


arr1 = np.arange(1,17).reshape(2,2,4).astype(np.float64)


# 33. Swap axes of the array you created in Question 32

# In[226]:


print(arr1.T)


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[238]:


ar1 = np.sqrt(np.arange(10))
ar1[ar1 < 0.5] =0


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[244]:


r1 = np.random.randn(12)
r2 = np.random.randn(12)
array1 = np.array((np.max(r1),np.max(r2)))


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[245]:


np.unique(names)


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[289]:


a1 = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
np.delete(a1,a1.index(np.intersect1d(a1,b)),axis =0)


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[273]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
newsampleArray = np.delete(sampleArray,2 ,axis=0)
newColumn = np.array([[10,10,10]])
np.append(newsampleArray,newColumn,axis = 0)


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[254]:


x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
np.dot(x,y)


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[260]:


np.random.randn(21).cumsum()

