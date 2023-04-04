import numpy as np

def svd(data,k=0):
    u,i,v = np.linalg.svd(data)
    if k>0:
        u=u[:,0:k]
        i=np.diag(i[0:k])
        v=v[0:k,:]
    return u,i,v

def predictSingle(u_index,i_index,u,i,v):
    return u[u_index].dot(i).dot(v.T[i_index].T)

def play():
    k=3
    #假设用户物品共现矩阵如下
    data = np.mat([[1,2,3,1,1],[1,3,3,1,2],[3,1,1,2,1],[1,2,3,3,1]])
    u,i,v = svd(data,k) # k
    # print(u.shape)
    # print(v.shape)
    # print(data)
    # print(u @ np.concatenate([np.diag(i),np.zeros((u.shape[0],1))],axis=-1) @ v)
    # print('-'*20)
    
    # k截断矩阵分解
    print(u.dot(i).dot(v))

    print(predictSingle(0, 0, u, i, v))

if __name__ == '__main__':
    play()