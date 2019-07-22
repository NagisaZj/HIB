import numpy as np
import matplotlib.pyplot as plt



if __name__=="__main__":
    logdir = './data/mnist-2-instance1'
    filedir = logdir+'/data.npz'
    data = np.load(filedir)
    loss = data['loss']
    c_v = data['clear_var']
    v_v = data['vague_var']
    c_c=data['clear_confidence']
    v_c = data['vague_confidence']

    plt.figure()
    plt.plot(loss)
    plt.figure()
    plt.plot(c_v,label='clear var')
    plt.plot(v_v, label='vague var')
    plt.legend()
    plt.figure()
    plt.plot(c_c, label='clear confidence')
    plt.plot(v_c, label='vague confidence')
    plt.legend()

    plt.show()