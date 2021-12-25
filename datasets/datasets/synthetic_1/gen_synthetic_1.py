"""
Follow Hara & Hayashi defragTrees AISTAT18 method
Synthetic 1 - this is '4-squares' synthetic data, with 10% label noise
2000 samples
"""
import numpy as np
import pandas as pd


SEED = 2020
N_DATA = 2000
N_DIM = 2


if __name__ == '__main__':
    np.random.seed(SEED)
    x = np.random.rand(N_DATA, N_DIM)
    # y = np.zeros(N_DATA)
    y = np.logical_xor(x[:,0]>0.5, x[:,1]>0.5)
    y = np.logical_xor(y, np.random.rand(2000) > .9)
    y = y.reshape(-1,1)
    array = np.hstack([x,y])

    df = pd.DataFrame(array, columns=['x1','x2','label'])
    df['label'] = df['label'].astype(int)
    df.to_csv('./synthetic_1.csv.gz', index=False, compression='gzip')
