import numpy as np

def get_rankedduculist(scores):
    
    duculist =np.array([i for i in range(scores.shape[0])]).reshape(-1,1)
    
    doculist_score = np.append(duculist,scores,axis=1)
    
    rankedduculist  = (doculist_score[(-doculist_score[:,-1]).argsort()])[:,0]
    rankedduculist  = [int(i) for i in rankedduculist]
    
    return rankedduculist