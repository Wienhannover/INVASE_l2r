#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import torch.nn.functional as F


# ### 1  Actor,Critic,Baseline network¶

# In[11]:


activation_dict = {"relu": nn.ReLU(),"sigmoid": nn.Sigmoid(),"softmax": nn.Softmax(),"selu": nn.SELU()}

#Use feature as the input and output selection probability
class Actor(nn.Module):
    
    def __init__(self, input_dim, h_dim, output_dim, layer_num, activation):
        super(Actor, self).__init__()
        #add regularization term in loss in pytroch, not every layer in keras
        layer_list = []
        layer_list.append(nn.Linear(input_dim, h_dim))
        layer_list.append(activation_dict[activation])
        for _ in range(layer_num - 2):
            layer_list.append(nn.Linear(h_dim, h_dim))
            layer_list.append(activation_dict[activation])
        layer_list.append(nn.Linear(h_dim, output_dim))
        layer_list.append(activation_dict["sigmoid"])
        
        self.linears = nn.Sequential(*layer_list)
        
    def forward(self, x):
        return self.linears(x)
        
#Use selected feature as the input and predict labels    
class Critic_RankNet(nn.Module):
    def __init__(self, inputs, hidden_size, outputs):
        super(Critic_RankNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(inputs, hidden_size),
            #nn.Dropout(0.5),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2,  inplace=True),#inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出
            #nn.SELU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2,  inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2,  inplace=True),
            nn.Linear(hidden_size, outputs),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_1, selection_1, input_2, selection_2):
        
        input_1 = np.array(input_1) * np.array(selection_1)
        result_1 = self.model(torch.from_numpy(input_1)) #预测input_1得分
        
        input_2 = np.array(input_2) * np.array(selection_2)
        result_2 = self.model(torch.from_numpy(input_2)) #预测input_2得分
        
        pred = self.sigmoid(result_1 - result_2) #input_1比input_2更相关概率
        return pred

    def predict(self, input, selection):
        
        input = np.array(input) * np.array(selection)
        result = self.model(torch.from_numpy(input))
        return result   

#Use the original feature as the input and predict labels
class Baseline_RankNet(nn.Module):
    
    def __init__(self, inputs, hidden_size, outputs):
        super(Baseline_RankNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(inputs, hidden_size),
            #nn.Dropout(0.5),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2,  inplace=True),#inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出
            #nn.SELU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2,  inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2,  inplace=True),
            nn.Linear(hidden_size, outputs),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_1, input_2):
        
        result_1 = self.model(input_1) #预测input_1得分
        result_2 = self.model(input_2) #预测input_2得分
        pred = self.sigmoid(result_1 - result_2) #input_1比input_2更相关概率
        return pred

    def predict(self, input):
        result = self.model(input)
        return result   


# ### 2  data preparation

# In[12]:


def extract_features(toks):
    # 获取features
    features = []
    for tok in toks:
        features.append(float(tok.split(":")[1]))
    return features

def extract_query_data(tok):
    #获取queryid documentid
    query_features = [tok.split(":")[1]] #qid
    return query_features

def get_format_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            data, _, comment = line.rstrip().partition("#")
            toks = data.split()
            y_train.append(int(toks[0])) #相关度
            x_train.append(extract_features(toks[2:])) # doc features
            query_id.append(extract_query_data(toks[1])) #qid
            

def get_pair_doc_data(y_train, query_id):
    #两两组合pair
    pairs = []
    tmp_x0 = []
    tmp_y0 = []
    tmp_x1 = []
    tmp_y1 = []
    tmp_within_query_signal = []
    
    for i in range(0, len(query_id) - 1):
        
        within_query_signal = 0      
        for j in range(i + 1, len(query_id)):
            #belongs to same query
            if query_id[i][0] == query_id[j][0]:
                within_query_signal = 1
                
            pairs.append([i,j])
            tmp_x0.append(x_train[i])
            tmp_y0.append(y_train[i])
            tmp_x1.append(x_train[j])
            tmp_y1.append(y_train[j])
            tmp_within_query_signal.append(within_query_signal)

    array_train_x0 = np.array(tmp_x0)
    array_train_y0 = np.array(tmp_y0)
    array_train_x1 = np.array(tmp_x1)
    array_train_y1 = np.array(tmp_y1)
    array_within_query_signal = np.array(tmp_within_query_signal)
    
    samples = int(array_train_x0.shape[0] * samples_portion_of_all)
    sample_1 = int(samples / 2)
    sample_0 = samples - sample_1
    
    signal_bool = array_within_query_signal.astype(bool)
    
    array_train_x0_1 = array_train_x0[signal_bool]
    array_train_y0_1 = array_train_y0[signal_bool]
    array_train_x1_1 = array_train_x1[signal_bool]
    array_train_y1_1 = array_train_y1[signal_bool]
    array_within_query_signal_1 = array_within_query_signal[signal_bool]
    
    sample_1_index = np.random.choice(np.arange(array_train_x0_1.shape[0]), sample_1, replace=False)
    
    tmp_array_train_x0_1 = array_train_x0_1[sample_1_index]
    tmp_array_train_y0_1 = array_train_y0_1[sample_1_index]
    tmp_array_train_x1_1 = array_train_x1_1[sample_1_index]
    tmp_array_train_y1_1 = array_train_y1_1[sample_1_index]
    tmp_array_within_query_signal_1 = array_within_query_signal_1[sample_1_index]
    #-------------------------------------------------------------------------------------
    array_train_x0_0 = array_train_x0[(1-signal_bool).astype(bool)]
    array_train_y0_0 = array_train_y0[(1-signal_bool).astype(bool)]
    array_train_x1_0 = array_train_x1[(1-signal_bool).astype(bool)]
    array_train_y1_0 = array_train_y1[(1-signal_bool).astype(bool)]
    array_within_query_signal_0 = array_within_query_signal[(1-signal_bool).astype(bool)]
    
    sample_0_index = np.random.choice(np.arange(array_train_x0_0.shape[0]), sample_0, replace=False)

    tmp_array_train_x0_0 = array_train_x0_0[sample_0_index]
    tmp_array_train_y0_0 = array_train_y0_0[sample_0_index]
    tmp_array_train_x1_0 = array_train_x1_0[sample_0_index]
    tmp_array_train_y1_0 = array_train_y1_0[sample_0_index]
    tmp_array_within_query_signal_0 = array_within_query_signal_0[sample_0_index]
    
    #-----combine----
    
    new_array_train_x0 = np.vstack((tmp_array_train_x0_1, tmp_array_train_x0_0))
    new_array_train_y0 = np.hstack((tmp_array_train_y0_1, tmp_array_train_y0_0))
    new_array_train_x1 = np.vstack((tmp_array_train_x1_1, tmp_array_train_x1_0))
    new_array_train_y1 = np.hstack((tmp_array_train_y1_1, tmp_array_train_y1_0))
    new_array_within_query_signal = np.hstack((tmp_array_within_query_signal_1, tmp_array_within_query_signal_0))
        
    return samples, new_array_train_x0, new_array_train_y0, new_array_train_x1, new_array_train_y1, new_array_within_query_signal


# #### Dstaset


class Dataset(data.Dataset):

    def __init__(self, data_path):
        # 解析训练数据
        get_format_data(data_path)
        # pair组合
        self.datasize, self.array_train_x0, self.array_train_y0, self.array_train_x1, self.array_train_y1, self.array_within_query_signal = get_pair_doc_data(y_train, query_id)

    def __getitem__(self, index):
        
        data1 = torch.from_numpy(self.array_train_x0[index]).float()
        y1 = torch.tensor(self.array_train_y0[index]).float()
        
        data2 = torch.from_numpy(self.array_train_x1[index]).float()
        y2 = torch.tensor(self.array_train_y1[index]).float()
        
        signal = torch.tensor(self.array_within_query_signal[index]).float()
        
        return data1, y1, data2, y2, signal

    def __len__(self):
        return self.datasize

def get_loader(data_path, batch_size, shuffle, drop_last):
    
    dataset = Dataset(data_path)
    
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last=drop_last
    )
    return data_loader


# ### Definition of Loss and Training process

# In[14]:


def pair_actor_loss(actor_output_1, actor_output_2, selection_1, selection_2, critic_loss_1, critic_loss_2, baseline_loss_1, baseline_loss_2, signal, lamda, beta, gamma):

    m = torch.nn.Softmax(dim=1)
    
    Reward_1 = critic_loss_1 - baseline_loss_1
    Pi_1 = (selection_1 * torch.log(actor_output_1 + 1e-8) + (1-selection_1) * torch.log(1-actor_output_1 + 1e-8)).sum(1)
    L0_1 = actor_output_1.detach().mean(1)
#     custom_actor_loss_1 = Pi_1 * Reward_1 + lamda * L0_1
#     L0_1 = selection_1.mean(1)
    custom_actor_loss_1 = Pi_1 * Reward_1 + lamda * L0_1
    #*************************************************************************
    Reward_2 = critic_loss_2 - baseline_loss_2
    Pi_2 = (selection_2 * torch.log(actor_output_2 + 1e-8) + (1-selection_2) * torch.log(1-actor_output_2 + 1e-8)).sum(1)
    L0_2 = actor_output_2.detach().mean(1)
#     custom_actor_loss_2 = Pi_2 * Reward_2 + lamda * L0_2
#     L0_2 = selection_2.mean(1)
    custom_actor_loss_2 = Pi_2 * Reward_2 + lamda * L0_2
    #***************************************************************************
    
    actor_output_1 = m(actor_output_1)
    actor_output_2 = m(actor_output_2)
    
    selection_loss = -((actor_output_1 * torch.log(actor_output_2 + 1e-8) + (1-actor_output_1) * torch.log(1-actor_output_2 + 1e-8))).sum(1)

#     final_selection_loss = beta * (selection_loss * signal) - gamma * (selection_loss * (1 - signal))

#     print(selection_loss.shape)
#     print(signal.shape)
#     print("-------------")
    
    signal_beta = signal.type(torch.bool)
    signal_gamma = (1 - signal).type(torch.bool)
    final_selection_loss = torch.FloatTensor(selection_loss.size()).type_as(selection_loss)
    
    final_selection_loss[signal_beta] = beta * selection_loss[signal_beta]
    final_selection_loss[signal_gamma] = -gamma * selection_loss[signal_gamma]
    
    
    return ((custom_actor_loss_1.mean() + custom_actor_loss_2.mean())/2) + final_selection_loss.mean()


# In[15]:


def train_model(baseline_model, actor_model,critic_model, epochs, lamda, beta, gamma):
        
    actor_optimizer = torch.optim.Adam(actor_model.parameters(),lr = 1e-5, weight_decay=1e-5)
    critic_optimizer = torch.optim.Adam(critic_model.parameters(),lr = 1e-4, weight_decay=1e-5)
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(),lr = 1e-4, weight_decay=1e-5)
    
    critic_criterion = nn.BCELoss()
    baseline_criterion = nn.BCELoss()
    loss_criterion = nn.BCELoss()
    #restrict_f = nn.Sigmoid()
    
    for epoch in range(epochs):
        
        epoch_train_actor_loss_output = []
        
        actor_model.train()
        critic_model.train()
        baseline_model.train()

        for batch, (data1, y1, data2, y2, signal) in enumerate(train_loader):
            
            # get selections of data1 and data2
            actor_output_1 = actor_model(data1.float())
            selection_1 = torch.bernoulli(torch.tensor(actor_output_1))
            
            actor_output_2 = actor_model(data2.float())
            selection_2 = torch.bernoulli(torch.tensor(actor_output_2))

            # train critic model
            critic_output = critic_model(data1.float(), selection_1, data2.float(), selection_2)
            
            label_difference = y1.ge(y2).double()
            critic_loss_output = critic_criterion(critic_output.double(), label_difference)

            critic_optimizer.zero_grad()
            critic_loss_output.backward()
            critic_optimizer.step()
            
            critic_output_1 = critic_model.predict(data1.float(), selection_1)
            critic_output_2 = critic_model.predict(data2.float(), selection_2)
            
            # train basseline model
            baseline_output = baseline_model(data1.float(), data2.float())
            baseline_loss_output = baseline_criterion(baseline_output.double(), label_difference)

            baseline_optimizer.zero_grad()
            baseline_loss_output.backward()
            baseline_optimizer.step()
            
            baseline_output_1 = baseline_model.predict(data1.float())
            baseline_output_2 = baseline_model.predict(data2.float())

            critic_loss_1 = -((y1.ge(1).float().view(-1,1) * torch.log(critic_output_1 + 1e-8)) + (1-y1.ge(1).float().view(-1,1)) * torch.log(1 - critic_output_1 + 1e-8))
            critic_loss_2 = -((y2.ge(1).float().view(-1,1) * torch.log(critic_output_2 + 1e-8)) + (1-y2.ge(1).float().view(-1,1)) * torch.log(1 - critic_output_2 + 1e-8))
                        
            baseline_loss_1 = -((y1.ge(1).float().view(-1,1) * torch.log(baseline_output_1 + 1e-8)) + (1-y1.ge(1).float().view(-1,1)) * torch.log(1 - baseline_output_1 + 1e-8))
            baseline_loss_2 = -((y2.ge(1).float().view(-1,1) * torch.log(baseline_output_2 + 1e-8)) + (1-y2.ge(1).float().view(-1,1)) * torch.log(1 - baseline_output_2 + 1e-8))
        
            # update selector network
            actor_loss_output = pair_actor_loss(actor_output_1, actor_output_2, selection_1, selection_2, critic_loss_1, critic_loss_2, baseline_loss_1, baseline_loss_2, signal, lamda, beta, gamma)
                        
            actor_optimizer.zero_grad()
            actor_loss_output.backward()
            actor_optimizer.step()
            
            epoch_train_actor_loss_output.append(actor_loss_output.item())
            
        print(epoch+1,"***********************************************************************")
        print("---------------train actor loss-------------", np.mean(epoch_train_actor_loss_output))
            
        epoch_vali_actor_loss_output = []
        epoch_vali_critic_acc = []
        
        actor_model.eval()
        critic_model.eval()
        baseline_model.eval()  
        
        with torch.no_grad():   
            for batch, (data1, y1, data2, y2, signal) in enumerate(vali_loader):
                                
                vali_actor_output_1 = actor_model(data1.float())
                vali_selection_1 = torch.bernoulli(torch.tensor(vali_actor_output_1))
                vali_actor_output_2 = actor_model(data2.float())
                vali_selection_2 = torch.bernoulli(torch.tensor(vali_actor_output_2))

                vali_critic_output_1 = critic_model.predict(data1.float(), vali_selection_1)
                vali_critic_output_2 = critic_model.predict(data2.float(), vali_selection_2)
                vali_baseline_output_1 = baseline_model.predict(data1.float())
                vali_baseline_output_2 = baseline_model.predict(data2.float())
                
                
                vali_critic_loss_1 = -((y1.ge(1).float().view(-1,1) * torch.log(vali_critic_output_1 + 1e-8)) + (1-y1.ge(1).float().view(-1,1)) * torch.log(1 - vali_critic_output_1 + 1e-8))
                vali_critic_loss_2 = -((y2.ge(1).float().view(-1,1) * torch.log(vali_critic_output_2 + 1e-8)) + (1-y2.ge(1).float().view(-1,1)) * torch.log(1 - vali_critic_output_2 + 1e-8))
                vali_baseline_loss_1 = -((y1.ge(1).float().view(-1,1) * torch.log(vali_baseline_output_1 + 1e-8)) + (1-y1.ge(1).float().view(-1,1)) * torch.log(1 - vali_baseline_output_1 + 1e-8))
                vali_baseline_loss_2 = -((y2.ge(1).float().view(-1,1) * torch.log(vali_baseline_output_2 + 1e-8)) + (1-y2.ge(1).float().view(-1,1)) * torch.log(1 - vali_baseline_output_2 + 1e-8))
        
                vali_actor_loss_output = pair_actor_loss(vali_actor_output_1, vali_actor_output_2, vali_selection_1, vali_selection_2, vali_critic_loss_1, vali_critic_loss_2, vali_baseline_loss_1, vali_baseline_loss_2, signal, lamda, beta, gamma)

                epoch_vali_actor_loss_output.append(vali_actor_loss_output)                
            
        print("---------------Vali actor loss-------------", np.mean(epoch_vali_actor_loss_output))
        #print("---------------Critic Accuracy-------------", np.mean(epoch_vali_critic_acc))

    return actor_model,critic_model,baseline_model





model_para = {'lambda':0.3,
              'actor_h_dim':300,
              'critic_h_dim':200,
              'baseline_h_dim':200,
              'actor_output' :46,
              'critic_output':1,
              'baseline_output':1,
              'n_layer':3,
              'activation':'selu',
              'learning_rate':0.0001}
batch_size = 32


# In[17]:


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


# In[18]:


actor_list = []
critic_list = []
baseline_list = []

beta = 0.1
gamma = 0.5
samples_portion_of_all = 0.0001

for k in range(1):

    y_train = []
    x_train = []
    query_id = []
    array_train_x1 = []
    array_train_x0 = []

    path = "./MQ2008/Fold{}/".format(k+1)

    train_path = path + 'train.txt'
    train_loader = get_loader(train_path, batch_size, shuffle=True, drop_last=True)

    vali_path = path + 'vali.txt'
    vali_loader = get_loader(vali_path, batch_size, shuffle=True, drop_last=True)

    test_path = path + 'test.txt'
    test_loader = get_loader(test_path, batch_size, shuffle=True, drop_last=True)

    actor = Actor(46, model_para['actor_h_dim'], model_para['actor_output'], model_para['n_layer'], model_para['activation'])
    critic = Critic_RankNet(46, model_para['critic_h_dim'], model_para['critic_output'])
    baseline = Baseline_RankNet(46, model_para['baseline_h_dim'], model_para['baseline_output'])

    actor.apply(init_weights)
    critic.apply(init_weights)
    baseline.apply(init_weights)
    
    trained_model_list = train_model(baseline, actor, critic, 300, model_para['lambda'], beta, gamma)

    actor_list.append(trained_model_list[0])
    critic_list.append(trained_model_list[1])
    baseline_list.append(trained_model_list[2])


# ### save

# In[ ]:


for k in range(1):    
    torch.save(actor_list[k].state_dict(), './tmp_model_saved/sample_0.0001_of_all_balance_model.train_initialize_weight/***_(beta_0.1_gamma_0.5_epoch_1000_batch_32)_actor_{}.pth'.format(k))
    torch.save(critic_list[k].state_dict(), './tmp_model_saved/sample_0.0001_of_all_balance_model.train_initialize_weight/***_(beta_0.1_gamma_0.5_epoch_1000_batch_32)_critic_{}.pth'.format(k))
    torch.save(baseline_list[k].state_dict(), './tmp_model_saved/sample_0.0001_of_all_balance_model.train_initialize_weight/***_(beta_0.1_gamma_0.5_epoch_1000_batch_32)_baseline_{}.pth'.format(k))





