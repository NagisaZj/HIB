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
def sample_batch(data_train,label_train,random_ratio,batch_size,vague_ratio,num_per_class):
    batch = np.zeros((batch_size,28,28*2))
    labels = np.zeros((batch_size,1))
    random_num = int(batch_size*random_ratio)
    vague_num = int(batch_size * vague_ratio)
    task_indices1 = np.random.randint(0,100,(random_num,))
    task_indices2 = np.random.randint(0, 100, (random_num,))
    good_indices = np.random.randint(0, 100, (batch_size-random_num,))
    task_indices1 = np.hstack((task_indices1,good_indices))
    task_indices2 = np.hstack((task_indices2,good_indices))
    labels[:,0] = (task_indices1==task_indices2)
    buffer_indices1 = np.random.randint(0,num_per_class,(batch_size,))
    buffer_indices2 = np.random.randint(0, num_per_class, (batch_size,))
    idx1 = buffer_indices1 + task_indices1 * num_per_class
    idx2 = buffer_indices2 + task_indices2 * num_per_class
    batch1 = data_train[idx1,:,:]
    batch2 = data_train[idx2,:,:]
    batch1, batch2 = make_vague(batch1,vague_num), make_vague(batch2,vague_num)
    return torch.cuda.FloatTensor(batch1), torch.cuda.FloatTensor(batch2), torch.cuda.FloatTensor(labels)

def make_vague(batch,vague_num):
    idx = np.random.choice(batch.shape[0],vague_num,replace=False)
    vague_len = np.random.randint(0, 28, (vague_num,))
    vague_x = np.array(np.floor(np.random.rand(vague_num) * (vague_len * -1 + 28)), dtype=np.int)
    vague_y = np.array(np.floor(np.random.rand(vague_num) * (vague_len * -1 + 28 * 2)), dtype=np.int)
    for i in range(vague_num):
        batch[idx[i], vague_x[i]:vague_x[i] + vague_len[i], vague_y[i]:vague_y[i] + vague_len[i]] = 0

    return batch



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
    model = HIB_model_cnn(beta=1e-4)
    batch_size = 128
    random_ratio = 0.6
    vague_ratio = 0.2
    data_train_clear,label_train_clear = load_data('./n-digit-mnist/data/dataset_mnist_2_instance/train-clear.npz')
    data_train_vague, label_train_vague = load_data('./n-digit-mnist/data/dataset_mnist_2_instance/train-vague.npz')
    data_test_clear, label_test_clear = load_data('./n-digit-mnist/data/dataset_mnist_2_instance/test-clear.npz')
    data_test_vague, label_test_vague = load_data('./n-digit-mnist/data/dataset_mnist_2_instance/test-vague.npz')

    data_train, label_train = load_data('./n-digit-mnist/data/dataset_mnist_2_instance/train.npz')
    data_test, label_test = load_data('./n-digit-mnist/data/dataset_mnist_2_instance/test.npz')

    num_per_class = int(data_train.shape[0]/100)

    for episode in range(500000):
        train_data1,train_data2,train_label = sample_batch(data_train,label_train,random_ratio,batch_size,vague_ratio,num_per_class)
        loss = model.cal_loss(train_data1,train_data2,train_label)
        model.optimize(loss)
        print(loss.cpu().data.numpy())
