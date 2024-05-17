#!/usr/bin/env python
# coding: utf-8

# In[45]:


x = (2,2,2,2,4,7,7,7,7,13,13,13,13,13,13,13,13,17,17,33,33,33,33,33,33,33,33,33
     ,33,33,33,33,33,33,33,33,33,33,33,33,33,33,41,41,41,41,41,41,41)

def ascii_density_histogram(val):
        histogram = {}
        for p in val:
            histogram[p] = histogram.get(p, 0) + 1
        return histogram
    
    
def ascii_histogram(val):
    cnt = ascii_density_histogram(val)
    for p in sorted(cnt):
        print('{0:5d} {1}'.format(p, '#' * cnt[p]))
        


print(ascii_density_histogram(x))

ascii_histogram(x)


# In[ ]:




