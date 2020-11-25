import numpy as np
import torch

# group data by query id
def group_data(txt_path, dataset):
    test_content = np.genfromtxt("{}{}.txt".format(txt_path, dataset),dtype=np.dtype(str))

    #data in test set grouped by qid
    x_y = {}
    for i in range(test_content.shape[0]):
        qid = np.int(test_content[i][1][4:])

        features = []
        for j in range(2, test_content.shape[1]):
            features.append(np.float(test_content[i][j][-8:]))

        #labels as last column
        label = np.float(test_content[i][0])
#         if(label > 1):
#             label = 1  
        features.append(label)
        #注：原始文档中label 为0，1，2 


        if qid in x_y.keys():
            x_y[qid].append(features)
        else:
            x_y[qid] = []
            x_y[qid].append(features)
    for key in x_y.keys():
        x_y[key] = torch.tensor(x_y[key])
        
    return x_y


    
    


