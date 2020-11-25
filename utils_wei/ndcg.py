import numpy as np

def dcg_score(y_score, k):
    
    y_score = np.array(y_score)
    y_score = y_score[:k]

    gains = 2**y_score - 1    
    discounts = np.log2(np.arange(k) + 2)
    dcg = np.sum(gains/discounts)
    
    return dcg
    
def ndcg_score(y_true, y_score, k=10):
    
    if k > len(y_true) or k > len(y_score):
        k = min(len(y_true),len(y_score))
    
    # retured results from neural network should be ranked as a list for one query, then retured to user
    # y_score_ranked = -np.sort(-y_score)
    # corresponding true label
    y_true_correspond = y_true[np.argsort(-y_score)]
    # ideal rank
    y_true_ideal = -np.sort(-y_true)
    
    dcg = dcg_score(y_true_correspond, k)
    idcg = dcg_score(y_true_ideal, k)
    
    ##if all documents for one query are originally irrelavant,then dcg score is 0
    if idcg == 0:
        return None
    return dcg/idcg


def dcg_score_linear(y_score, k):
    
    y_score = np.array(y_score)
    y_score = y_score[:k]

    gains = y_score    
    discounts = np.log2(np.arange(k) + 2)
    dcg = np.sum(gains/discounts)
    
    return dcg
    
def ndcg_score_linear(y_true, y_score, k=10):
    
    if k > len(y_true) or k > len(y_score):
        k = min(len(y_true),len(y_score))
    
    y_true_correspond = y_true[np.argsort(-y_score)]

    y_true_ideal = -np.sort(-y_true)
    
    dcg = dcg_score_linear(y_true_correspond, k)
    idcg = dcg_score_linear(y_true_ideal, k)
    
    if idcg == 0:
        return None
    return dcg/idcg
    
if __name__ == '__main__':
    a =np.array([3,2,3,0,1,2])
    b =np.array([3,3,3,2,2,1])
    print(ndcg_score(b,a))
    print(dcg_score(a,6))
    print(dcg_score(b,6))
    print(dcg_score([0,0,0,0,0,0,0,0,0],6))

    