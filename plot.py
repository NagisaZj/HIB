import matplotlib.pyplot as plt
import numpy as np



if __name__=="__main__":
    loss = np.array(np.load('./data/loss.npy'),dtype=np.float32)
    good_confidence = np.array(np.load('./data/good_confidence.npy'), dtype=np.float32)
    vague_confidence = np.array(np.load('./data/vague_confidence.npy'), dtype=np.float32)
    good_std = np.array(np.load('./data/good_std.npy'), dtype=np.float32)
    vague_std = np.array(np.load('./data/vague_std.npy'), dtype=np.float32)

    plt.figure()
    plt.plot(loss)
    plt.title('loss')

    plt.figure()
    plt.plot(good_confidence,'g',label='clear data confidence')
    plt.plot(vague_confidence,'r',label='vague data confidence')
    plt.legend()
    plt.title('confidence')

    plt.figure()
    plt.plot(good_std, 'g', label='clear data std')
    plt.plot(vague_std, 'r', label='vague data std')
    plt.legend()
    plt.title('std of z|x')


    plt.show()