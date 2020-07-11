import numpy as np 
import pdb

ar=np.array([[1,2],[3,4],[5,6]])

new_ar=np.take(ar,[0,2],axis=0)
print (new_ar)
pdb.set_trace()