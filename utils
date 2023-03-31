import numpy as np
import math

def conv1d(data, kernel, stride=1):
    step = int(math.floor(data.shape[1]/stride-kernel.shape[1]+1))
    new_data = np.zeros(step)
    for i in range(step):
        ele = np.sum(data[:,i:i+int(kernel.shape[1])]*kernel)
        new_data[i]=ele
        
    return new_data,step
