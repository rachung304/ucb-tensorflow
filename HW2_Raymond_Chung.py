
# coding: utf-8

# ## Homework 2
# ### Name: Raymond Chung
# ### ID: X117634

# In[1]:


import tensorflow as tf
import numpy as np


# In[2]:


sess = tf.Session()


# In[3]:


# Placeholder for an input array with dtype float32 and shape None
with tf.name_scope("Input_placeholder"):
    a = tf.placeholder(tf.int32, shape=None, name="input_a")


# In[4]:


# Scopes for the input, middle section and final node f
with tf.name_scope("Middle_section"):
    b = tf.reduce_prod(a, name="prod_b")
    c = tf.reduce_mean(a, name="mean_c")
    d = tf.reduce_sum(a, name="sum_d")
    e = tf.add(c, b, name="add_e")

with tf.name_scope("Final_node"):
    f = tf.multiply(e, d, name="mul_f")


# In[5]:


# Feed the placeholder with an array A consis=ng of 100 normally distributed random numbers with
# mean = 1 and standard deviation = 2
input_dict = {a: np.random.normal(1, 2, 100)}


# In[6]:


# Save your graph and show it in TensorBoard
writer = tf.summary.FileWriter('./name_scope_HW2', graph=tf.get_default_graph())
writer.close()


# In[7]:


sess.run(a, feed_dict=input_dict)

## Plot you input array on a separate figure

plt.plot(sess.run(a, feed_dict=input_dict))
plt.show()

## Random input Array a
"""
array([ 4,  2,  0,  3, -2, -1,  0,  0,  2,  2,  1,  4,  1,  0, -1,  1,  0,
        1,  1,  0, -2,  0,  0,  1,  1,  0,  0,  1,  0,  2, -2, -1,  0,  1,
       -3,  0,  3,  0, -3, -1,  0,  0,  2,  1,  2,  0,  1, -1,  4,  2,  3,
       -1,  2, -1,  1, -2, -4,  2,  4,  1,  1,  0,  5,  2, -1, -1,  0,  3,
        1,  0,  1,  2,  2,  0,  1,  0,  4,  2,  4,  4, -1,  1,  2,  3,  2,
        0,  0,  0,  4,  1, -1,  0,  1,  0,  0,  0, -1,  0,  3,  3], dtype=int32)
"""
