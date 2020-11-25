from scipy.stats import kendalltau



def tau_valid(rank_original, rank_reproduced):
    
    kendall_tau, p_value = kendalltau(rank_original, rank_reproduced)
    
    return kendall_tau, p_value


#rank_reproduced is generated from complement feature set
def tau_completeness(rank_original, rank_reproduced):
    
    kendall_tau, p_value = kendalltau(rank_original, rank_reproduced)
    
    return -kendall_tau, p_value