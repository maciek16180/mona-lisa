import numpy as np

# crossover operators for ES
# parents is (num_pairs, 2, num_figs * fig_len * 2)

def cx_one_point(parents, num_figs, fig_len):
    num_pairs = parents.shape[0]
    parents = parents.reshape(num_pairs, 2, 2, num_figs, fig_len)
    
    points = np.random.choice(num_figs, size=num_pairs)
    children = np.empty((num_pairs * 2, 2, num_figs, fig_len))
    
    for i in xrange(num_pairs):
        p1, p2 = parents[i]
        cp = points[i]
        children[2 * i][:, :cp] = p1[:, :cp]
        children[2 * i][:, cp:] = p2[:, cp:]
        children[2 * i + 1][:, :cp] = p2[:, :cp]
        children[2 * i + 1][:, cp:] = p1[:, cp:]
        
    return children.reshape(num_pairs * 2, 2 * num_figs * fig_len)

def cx_random(parents, num_figs, fig_len):    
    num_pairs = parents.shape[0]
    
    parents = parents.reshape(num_pairs, 2, 2, num_figs, fig_len).transpose(0,1,3,2,4)
    parents = parents.reshape(num_pairs, 2 * num_figs, fig_len * 2)
    points = np.random.rand(num_pairs, num_figs) < .5
    
    indsL = np.indices((num_pairs, num_figs))[1]
    indsR = np.indices((num_pairs, num_figs))[1]
    
    indsL[points == 1] += num_figs
    indsR[points == 0] += num_figs
    
    inds_columns = np.vstack([indsL, indsR])    
    inds_rows = np.indices((2, num_pairs))[1].flatten()[:, np.newaxis]
    
    children = parents[inds_rows, inds_columns]
    children = children.reshape(num_pairs * 2, num_figs, 2, fig_len).transpose(0,2,1,3)
    
    return children.reshape(num_pairs * 2, num_figs * fig_len * 2)