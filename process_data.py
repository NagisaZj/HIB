import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from hib_model import HIB_model_cnn
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import random
from PIL import Image
def sample_batch(generator,batch_size):
    batch = np.zeros((batch_size,5,10,2))
    labels = np.zeros((batch_size,1))
    random_num = int(batch_size*0.6)
    task_indices1 = np.random.randint(0,generator.num_tasks,(random_num,))
    task_indices2 = np.random.randint(0, generator.num_tasks, (random_num,))
    good_indices = np.random.randint(0, generator.num_tasks, (batch_size-random_num,))
    task_indices1 = np.hstack((task_indices1,good_indices))
    task_indices2 = np.hstack((task_indices2,good_indices))
    labels[:,0] = (task_indices1==task_indices2)
    buffer_indices1 = np.random.randint(0,generator.buffer_size,(batch_size,))
    buffer_indices2 = np.random.randint(0, generator.buffer_size, (batch_size,))
    batch[:,:,:,0] = generator.buffer[task_indices1,buffer_indices1,:,:]
    batch[:, :, :, 1] = generator.buffer[task_indices2, buffer_indices2, :, :]
    return batch, labels


def sample_test_batch(data,batch_size):
    num_tasks = data.shape[0]
    buffer_size = data.shape[1]
    good_num = int(buffer_size*0.8)
    task_label_good = np.random.randint(0,num_tasks,(batch_size,))
    task_label_bad = np.random.randint(0,num_tasks,(batch_size,))
    buffer_label_good = np.random.randint(0,good_num,(batch_size,))
    buffer_label_bad = np.random.randint(good_num, buffer_size, (batch_size,))
    good_batch = data[task_label_good,buffer_label_good,:,:]
    bad_batch = data[task_label_bad,buffer_label_bad,:,:]
    return good_batch,bad_batch


def visualize(mean_good,var_good,mean_vague,var_vague,num):
    mean_good,var_good,mean_vague,var_vague = mean_good.cpu().data.numpy(),var_good.cpu().data.numpy(),mean_vague.cpu().data.numpy(),var_vague.cpu().data.numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(mean_good.shape[0]):
        ell1 = Ellipse(xy=(mean_good[i,0], mean_good[i,1]), width=var_good[i,0], height=var_good[i,1], angle=0, facecolor='green', alpha=0.3)
        ell2 = Ellipse(xy=(mean_vague[i,0], mean_vague[i,1]), width=var_vague[i,0], height=var_vague[i,1], angle=0, facecolor='red', alpha=0.3)
        ax.add_patch(ell1)
        ax.add_patch(ell2)
        plt.plot(mean_good[i,0],mean_good[i,1],'g')
        plt.plot(mean_vague[i, 0], mean_vague[i, 1], 'r')
    plt.savefig("./data/figs/%i.png"%num)
    plt.close()
    return

def load_data(dir):
    data = np.load(dir)
    labels = data['labels']
    images = data['images']
    return images, labels

if __name__=="__main__":
    #model = HIB_model_cnn(beta=1e-4)
    data_train,label_train = load_data('./n-digit-mnist/data/dataset_mnist_2_instance/test.npz')
    random.seed(1337)
    num_data = data_train.shape[0]
    num_vague_data=int(num_data/4)
    #idx = np.random.permutation(num_data)
    #data_train,label_train = data_train[idx,...], label_train[idx,...]
    batch_size = 128

    vague_idx = np.random.choice(num_data,num_vague_data,replace=False)
    data_train_vague ,label_train_vague = data_train[vague_idx,...], label_train[vague_idx,...]

    vague_len = np.random.randint(0,28,(num_vague_data,))
    vague_x = np.array(np.floor(np.random.rand(num_vague_data) * (vague_len*-1+28)),dtype=np.int)
    vague_y = np.array(np.floor(np.random.rand(num_vague_data) * (vague_len*-1+28*2)),dtype=np.int)
    print(vague_len[:10],vague_x[:10],vague_y[:10])
    for i in range(num_vague_data):
        data_train_vague[i,vague_x[i]:vague_x[i]+vague_len[i],vague_y[i]:vague_y[i]+vague_len[i]] = 0

    print(label_train_vague[1])
    np.savez('./n-digit-mnist/data/dataset_mnist_2_instance/test-clear.npz',images=data_train,labels=label_train)
    np.savez('./n-digit-mnist/data/dataset_mnist_2_instance/test-vague.npz', images=data_train_vague, labels=label_train_vague)