# # Numpy_Assignment_2::

# ## Question:1

# ### Convert a 1D array to a 2D array with 2 rows?

# #### Desired output::

# array([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])

# In[4]:


import numpy as np
oneD_toD = np.arange(10).reshape(2,5)


# ## Question:2

# ###  How to stack two arrays vertically?

# #### Desired Output::
'''array([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])
       '''
# In[31]:


a = np.arange(10).reshape(2,5)
b= np.ones((2,5),dtype=int)
stack = np.vstack((a,b))


# ## Question:3

# ### How to stack two arrays horizontally?

# #### Desired Output::
'''array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
       [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])'''
# In[27]:


a = np.arange(10).reshape(2,5)
b= np.ones((2,5),dtype=int)
np.hstack((a,b))


# ## Question:4

# ### How to convert an array of arrays into a flat 1d array?

# #### Desired Output::
#array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# In[36]:


array_of_array = np.arange(20).reshape(4,5)
array_of_array.flatten()


# ## Question:5

# ### How to Convert higher dimension into one dimension?

# #### Desired Output::
#array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# In[38]:


array_of_array = np.arange(15).reshape(3,5)
array_of_array.ravel()


# ## Question:6

# ### Convert one dimension to higher dimension?

# #### Desired Output::
'''array([[ 0, 1, 2],
[ 3, 4, 5],
[ 6, 7, 8],
[ 9, 10, 11],
[12, 13, 14]])'''
# In[58]:


two_D=np.arange(15).reshape(5,3)


# ## Question:7

# ### Create 5x5 an array and find the square of an array?

# In[62]:


a = np.arange(25).reshape(5,5)
np.square(a)


# ## Question:8

# ### Create 5x6 an array and find the mean?

# In[64]:


a = np.arange(30).reshape(5,6)
np.mean(a)


# ## Question:9

# ### Find the standard deviation of the previous array in Q8?

# In[65]:


np.std(a)


# ## Question:10

# ### Find the median of the previous array in Q8?

# In[66]:


np.median(a)


# ## Question:11

# ### Find the transpose of the previous array in Q8?

# In[68]:


print(a.T)


# ## Question:12

# ### Create a 4x4 an array and find the sum of diagonal elements?

# In[75]:


arr = np.ones((4,4))
np.diagonal(arr).sum()


# ## Question:13

# ### Find the determinant of the previous array in Q12?

# In[76]:


np.linalg.det(arr)


# ## Question:14

# ### Find the 5th and 95th percentile of an array?

# In[80]:


np.percentile(arr,5)
np.percentile(arr,95)


# ## Question:15

# ### How to find if a given array has any null values?

# In[85]:


np.isnan(arr)


# In[ ]:




