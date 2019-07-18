import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Data_generator():
    def __init__(self,num_tasks=9,buffer_size=10000,goal_radius = 0.2, episode_length = 10):
        #8000 clear, 2000 vague
        self.episode_length = episode_length
        self.clear_num = int(buffer_size*0.8)
        self.goal_radius = goal_radius
        self.vague_num = buffer_size-self.clear_num
        #self.vague_per_task = int(self.vague_num/(num_tasks-1))
        self.buffer_size = buffer_size
        self.num_tasks = num_tasks
        self.angles = np.linspace(0,np.pi,num_tasks)
        self.goals = np.zeros((self.num_tasks,2))
        self.goals[:,0] = np.cos(self.angles)
        self.goals[:,1] = np.sin(self.angles)
        self.buffer = np.zeros((self.num_tasks,self.buffer_size,5,self.episode_length)) #only (s,a,r)now, will add s' in the future
        self.labels = np.zeros((self.num_tasks,self.buffer_size,1,self.episode_length))
        for i in range(self.num_tasks):
            self.labels[i,:,:,:] = i
        self._state = np.zeros((2,))

    def cal_rew(self,task_num,state):
        x = state[0] - self.goals[task_num,0]
        y = state[1] - self.goals[task_num, 1]
        dis = (x **2 + y **2) **0.5
        if dis<self.goal_radius:
            return 1-dis
        else:
            return 0

    def step(self,action):
        action = np.clip(action,-0.1,0.1)
        self._state = self._state + action
        rews = np.zeros((self.num_tasks,))
        for i in range(self.num_tasks):
            rews[i] = self.cal_rew(i,self._state)
        return np.copy(self._state),rews

    def reset(self):
        self._state = np.zeros((2,))
        return self._state

    def cal_action(self,state,task_num):
        goal = self.goals[task_num,:]
        dir = goal - state
        return np.clip(dir,-0.1,0.1)

    def generate_samples(self):
        vague_pointers = np.zeros((self.num_tasks,),dtype=np.int)+self.clear_num
        pointer = 0
        for i in range(self.num_tasks):

            for j in range(self.clear_num):
                state = self.reset()
                trajectory = np.zeros((4,self.episode_length))
                rewards = np.zeros((self.num_tasks,self.episode_length))
                for k in range(self.episode_length):
                    action = self.cal_action(state,i)
                    state_,rews = self.step(action)
                    trajectory[:2,k] = state
                    trajectory[2:4,k] = action
                    rewards[:,k] = rews
                    state = state_

                #stor trajectories
                self.buffer[i,j,:,:] = np.vstack((trajectory,rewards[i,:]))
                if j<self.vague_num:
                    if pointer == i:
                        pointer = (pointer + 1) % self.num_tasks
                    while vague_pointers[pointer]>=self.buffer_size:
                        pointer = (pointer + 1) % self.num_tasks
                    #print(i, j, pointer, vague_pointers[pointer])
                    self.buffer[pointer, vague_pointers[pointer], :, :] = np.vstack((trajectory, rewards[pointer, :]))

                    vague_pointers[pointer] += 1
                    pointer = (pointer + 1) % self.num_tasks




if __name__=="__main__":
    gen = Data_generator(buffer_size=200)
    gen.generate_samples()
    np.save('data.npy',gen.buffer)
    print(gen.buffer[2,:,:,:10])





