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
import os

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

def make_vague_visual(batch,vague_num):
    idx = np.arange(vague_num)
    vague_len = np.random.randint(0, 28, (vague_num,))
    vague_x = np.array(np.floor(np.random.rand(vague_num) * (vague_len * -1 + 28)), dtype=np.int)
    vague_y = np.array(np.floor(np.random.rand(vague_num) * (vague_len * -1 + 28 * 2)), dtype=np.int)
    for i in range(vague_num):
        batch[idx[i], vague_x[i]:vague_x[i] + vague_len[i], vague_y[i]:vague_y[i] + vague_len[i]] = 0

    return batch

def sample_batch_visual(data_train,label_train,batch_size,vague_ratio):
    labels = np.zeros((batch_size,1))
    vague_num = int(batch_size * vague_ratio)
    idx1 = np.random.randint(0, data_train.shape[0], (batch_size,))
    batch1 = data_train[idx1,:,:]
    labels[:,0] = label_train [idx1]
    batch1 = make_vague_visual(batch1,vague_num)
    return torch.cuda.FloatTensor(batch1), torch.cuda.FloatTensor(labels), vague_num


def visualize(z_mean,z_var,vague_num,episode,save_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(z_mean.shape[0]):
        if i < vague_num:
            ell1 = Ellipse(xy=(z_mean[i,0], z_mean[i,1]), width=z_var[i,0], height=z_var[i,1], angle=0, facecolor='red', alpha=0.3)
        else:
            ell1 = Ellipse(xy=(z_mean[i,0], z_mean[i,1]), width=z_var[i,0], height=z_var[i,1], angle=0, facecolor='green', alpha=0.3)
        ax.add_patch(ell1)
    plt.savefig(save_dir+"/figs/%i.png"%episode)
    plt.close()
    return

def load_data(dir):
    data = np.load(dir)
    labels = data['labels']
    images = data['images']
    return images, labels

if __name__=="__main__":
    model = HIB_model_cnn(beta=1e-4,hard=True)
    batch_size = 128
    random_ratio = 0.6
    vague_ratio = 0.8
    '''data_train_clear,label_train_clear = load_data('./n-digit-mnist/data/dataset_mnist_2_instance/train-clear.npz')
    data_train_vague, label_train_vague = load_data('./n-digit-mnist/data/dataset_mnist_2_instance/train-vague.npz')
    data_test_clear, label_test_clear = load_data('./n-digit-mnist/data/dataset_mnist_2_instance/test-clear.npz')
    data_test_vague, label_test_vague = load_data('./n-digit-mnist/data/dataset_mnist_2_instance/test-vague.npz')'''

    data_train, label_train = load_data('./n-digit-mnist/data/dataset_mnist_2_instance/train.npz')
    data_test, label_test = load_data('./n-digit-mnist/data/dataset_mnist_2_instance/test.npz')

    save_dir='./data/mnist-2-instance-hard'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir+'/figs'):
        os.makedirs(save_dir+'/figs')

    if not os.path.exists(save_dir+'/models'):
        os.makedirs(save_dir+'/models')

    num_per_class = int(data_train.shape[0]/100)
    loss_rem = []
    clear_var_rem = []
    vague_var_rem = []
    clear_confidence_rem = []
    vague_confidence_rem = []
    for episode in range(500000):
        train_data1,train_data2,train_label = sample_batch(data_train,label_train,random_ratio,batch_size,vague_ratio,num_per_class)
        loss = model.cal_loss(train_data1,train_data2,train_label)
        model.optimize(loss)
        print(episode,loss.cpu().data.numpy())

        if (episode+1) % 10 ==0:
            visual_batch,visual_label, vague_num = sample_batch_visual(data_test,label_test,batch_size=64,vague_ratio=vague_ratio)
            z_mean, z_var = model.cal_z(visual_batch)
            z_var = torch.sqrt(z_var)
            clear_confidence, vague_confidence = model._confidence(z_mean[vague_num:,:],z_var[vague_num:,:]),model._confidence(z_mean[:vague_num,:],z_var[:vague_num,:])
            z_mean, z_var = z_mean.cpu().data.numpy(),z_var.cpu().data.numpy()
            clear_confidence, vague_confidence = clear_confidence.cpu().data.numpy(), vague_confidence.cpu().data.numpy()
            clear_var, vague_var = np.mean(z_var[vague_num:,:]), np.mean(z_var[:vague_num,:])
            clear_confidence, vague_confidence = np.mean(clear_confidence), np.mean(vague_confidence)
            print('episode:',episode,' loss:', loss.cpu().data.numpy(), ' clear var:',clear_var,' vague var:',vague_var,' clear confidence:', clear_confidence,' vague confidence:', vague_confidence )
            loss_rem.append(loss.cpu().data.numpy())
            clear_var_rem.append(clear_var)
            vague_var_rem.append(vague_var)
            clear_confidence_rem.append(clear_confidence)
            vague_confidence_rem.append(vague_confidence)

            if (episode+1) % 10 ==0:
                np.savez(save_dir+'/data.npz',loss=loss_rem,clear_var=clear_var_rem,vague_var=vague_var_rem,clear_confidence = clear_confidence_rem, vague_confidence=vague_confidence_rem)
                visualize(z_mean,z_var,vague_num,episode,save_dir)
                torch.save(model.state_dict(),save_dir+'/models/model%i.p'%episode)
                #device = torch.device('cuda')
                #model.load_state_dict(torch.load('./data/mnist-2-instance/models/model%i.p'%episode))
                #model.to(device)
