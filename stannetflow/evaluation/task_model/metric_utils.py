import scipy.stats
import pandas as pd

def JS(pk, qk, KL=None):
    if KL is None:
        avgk = [(x+y)/2 for x,y in zip(pk, qk)]
        JS = (scipy.stats.entropy(pk, avgk) + scipy.stats.entropy(qk, avgk)) / 2
    elif KL is True:
        JS = scipy.stats.entropy(pk, qk)
    elif KL is False:
        JS = scipy.stats.entropy(qk, pk)
    else:
        input("Error: JS param bug.")
    return JS

def gd(pd_data, bins=304):
    #values1 = list(np.log(pd_data.byt))
    values1 = list(pd_data)
    #print(values1)
    cats1 = pd.cut(values1, bins)
    pr1 = list(cats1.value_counts())
    # print(pr1)
    #pk = get_distribution(pr1)
    pk = get_distribution_with_laplace_smoothing(pr1)
    return pk

def get_distribution_with_laplace_smoothing(a_count):
    k = 1.0
    tot_k = len(a_count) * k
    sum_count = sum(a_count)
    p = []
    for component in a_count:
        adj_component = (component + k) / (sum_count + tot_k)
        p.append(adj_component)
    # print('laplace_smoothing:\n', len(a_count), sum(a_count), sum(p), max(p), min(p))
    return p

def get_distribution(a_count):
    sum_count = sum(a_count)
    p = []
    for component in a_count:
        adj_component = component / sum_count
        p.append(adj_component)
    # print('get_distribution:\n', len(a_count), sum(a_count), sum(p), max(p), min(p))
    return p
