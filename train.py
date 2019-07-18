import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from hib_model import HIB_model
from data_generator import Data_generator
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
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


if __name__=="__main__":
    model = HIB_model(beta=1e-3)
    generator = Data_generator(buffer_size=100)
    data = np.array(np.load('data.npy'),dtype=np.float32)
    generator.buffer = data
    print(generator.buffer.shape)
    batch_size = 16
    loss_rem = []
    good_confidence_rem = []
    vague_confidence_rem = []
    good_var_rem = []
    vague_var_rem = []
    for i in range(1000):
        batch, labels = sample_batch(generator,batch_size)
        batch, labels = torch.FloatTensor(batch).cuda(3),torch.FloatTensor(labels).cuda(3)
        loss = model.cal_loss(batch,labels)

        model.optimize(loss)

        good_batch, vague_batch =sample_test_batch(data,8)
        good_batch, vague_batch = torch.FloatTensor(good_batch).cuda(3), torch.FloatTensor(vague_batch).cuda(3)
        mean_good, var_good = model.cal_z(good_batch)
        mean_vague, var_vague = model.cal_z(vague_batch)
        var_good,var_vague = torch.sqrt(var_good), torch.sqrt(var_vague)
        good_confidence, vague_confidence = model.cal_confidence(good_batch), model.cal_confidence(vague_batch)
        #conf1, conf2 = model.cal_confidence()
        print(i, loss.cpu().data.numpy(),'good variance ',torch.mean(var_good,dim=0).cpu().data.numpy(),' vague variance ',torch.mean(var_vague,dim=0).cpu().data.numpy())
        print('good confidence', torch.mean(good_confidence,dim=0).cpu().data.numpy(),' vague confidence ',torch.mean(vague_confidence,dim=0).cpu().data.numpy())
        loss_rem.append(loss.cpu().data.numpy())
        good_confidence_rem.append(torch.mean(good_confidence,dim=0).cpu().data.numpy())
        vague_confidence_rem.append(torch.mean(vague_confidence, dim=0).cpu().data.numpy())
        good_var_rem.append(torch.mean(var_good).cpu().data.numpy())
        vague_var_rem.append(torch.mean(var_vague).cpu().data.numpy())
        #print(data[0,:2,:,:],data[0,-2:,:,:])
        if (i+0)%50==0:
            visualize(mean_good,var_good,mean_vague,var_vague,i)
            np.save('./data/loss.npy',loss_rem)
            np.save('./data/good_confidence.npy', good_confidence_rem)
            np.save('./data/vague_confidence.npy', vague_confidence_rem)
            np.save('./data/good_std.npy', good_var_rem)
            np.save('./data/vague_std.npy', vague_var_rem)



