import numpy as np
import pdb

def round_to_one(m):
    m = np.around(m,3)

    try:
        indices=np.where(m.sum(axis=1)>1)
        for element in indices[0]:
            sums = m[element].sum()
            dif = sums-1
            x = np.nonzero(m[element])[0][0]
            m[element][x]=m[element][x]-dif
    except:
        pass

    indices=np.where(m.sum(axis=1)<1)
    try:
        for element in indices[0]:
            sums = m[element].sum()
            dif = 1-sums
            x = np.nonzero(m[element])[0][0]
            m[element][x]=m[element][x]+dif

    except:
        pass

    return m

if __name__ == '__main__':
    mm=np.array([[0.8,0.19],[0.8,0.19]])
    ff=round_to_one(mm)
    pdb.set_trace()

