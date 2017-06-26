import numpy as np
from cost import draw_individual_list as draw_individual
from cost import cost
from matplotlib.pyplot import imshow
import random, pickle
from itertools import chain
from copy import deepcopy, copy
from time import time


class HC_dyn():
    
    def __init__(self, pop_size, num_children, ref_img, board='black', distr='triangular',
                 F='default', min_num_edges=3, max_num_edges=10, max_num_figs=50, mode='polygon_simple',
                 mut_add_pol=.002, mut_add_point=.001, mut_rem_point=.001, mut_rem_pol=.001, mut_move_pol=.002, seed=12345):
        
        assert distr in ['triangular', 'uniform']
        
        if F == 'default':
            F = cost
        
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed
            
        self.distr = distr
            
        self.mode = mode
        self.shape = ref_img.size
        
        self.board = board
        
        self.mut_add_pol = mut_add_pol
        self.mut_add_point = mut_add_point
        self.mut_rem_point = mut_rem_point
        self.mut_rem_pol = mut_rem_pol
        self.mut_move_pol = mut_move_pol
        
        if self.mode == 'polygon_simple':
            self.max_num_edges = max_num_edges
            self.min_num_edges = min_num_edges
        else:
            raise NotImplementedError
        self.max_num_figs = max_num_figs
        
        self.pop_size = pop_size
        self.num_children = num_children
        self.ref_img = ref_img
        self.ref_matrix = np.array(self.ref_img, dtype=np.int32)
        self.F = lambda x: -F(x, self.ref_img, mode=self.mode, ref_matrix=self.ref_matrix, board=self.board)
        
        self.log = []
        self.iterations_done = 0
        self.best_imgs = []
        self.time_elapsed = 0
        
        self.population = [([], []) for i in xrange(self.pop_size)] # starting individuals have 0 polygons
        self.scores = self._score_population(self.population)
        
        self.best_ind = self.population[self.scores.argmax()]
        self.best_ind_score = self.scores.max()
        
    def _crop(self, num, xy):
        return min(max(0, num), self.shape[xy])
        
    def _generate_figure(self):
        x, y = self._generate_point()
        pol = []
        for i in xrange(self.min_num_edges):
            pol.append([self._crop(x + random.randint(-3, 3), 0),
                        self._crop(y + random.randint(-3, 3), 1)])
        col = [random.randint(0, 255) for i in xrange(3)] + [random.randint(10, 60)]
        return pol, col
    
    def _generate_point(self):
        return [random.randint(0, self.shape[0]), random.randint(0, self.shape[1])]
    
    def _score_population(self, P):
        return np.array(map(self.F, P))
    
    def _mutate_population(self, P):
        return [self._mutate_individual(ind) for ind in P]
    
    def _mutate_individual(self, ind):
        pols, cols = ind
        num_pols = len(pols)
        copied = False
        
        if num_pols:
            mut_pol = random.randint(0, num_pols - 1)
            mut_axis = random.randint(0, 2 * len(pols[mut_pol]) + 3)
            
            if mut_axis < 4:
                col = copy(cols[mut_pol])
                if self.distr == 'triangular':
                    col[mut_axis] = int(random.triangular(0, col[mut_axis], 255))
                else:
                    col[mut_axis] = random.randint(0, 255)
                cols = cols[:mut_pol] + [col] + cols[mut_pol + 1:]
            else:
                xy = mut_axis % 2
                pol = map(copy, pols[mut_pol])
                if self.distr == 'triangular':
                    pol[mut_axis / 2 - 2][xy] = int(random.triangular(0, pol[mut_axis / 2 - 2][xy], self.shape[xy]))
                else:
                    pol[mut_axis / 2 - 2][xy] = random.randint(0, self.shape[xy])
                pols = pols[:mut_pol] + [pol] + pols[mut_pol + 1:]
            
            rem_point = sorted(np.where(np.random.rand(num_pols) < self.mut_rem_point)[0])
            for i in rem_point:
                pol = copy(pols[i])
                if len(pol) > self.min_num_edges:
                    idx = random.randint(0, len(pol) - 1)
                    del pol[idx]
                    pols = pols[:i] + [pol] + pols[i + 1:]
            
            add_point = sorted(np.where(np.random.rand(num_pols) < self.mut_add_point)[0])
            for i in add_point:
                pol = copy(pols[i])
                if len(pol) < self.max_num_edges:
                    idx = random.randint(0, len(pol) - 1)
                    idx_next = (idx + 1) % len(pol)
                    p1 = pol[idx]
                    p2 = pol[idx_next]
                    new_point = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
                    pol.insert(idx + 1, new_point)
                    pols = pols[:i] + [pol] + pols[i + 1:]
                    
            if random.random() < self.mut_rem_pol:
                idx = random.randint(0, num_pols - 1)
                pols = pols[:idx] + pols[idx + 1:]

            if random.random() < self.mut_move_pol:
                idx1 = random.randint(0, len(pols) - 1)
                idx2 = random.randint(0, len(pols) - 1)
                pols, cols = copy(pols), copy(cols)               
                pol, col = pols.pop(idx1), cols.pop(idx2)
                pols.insert(idx2, pol)
                cols.insert(idx2, col)
                        
        if len(pols) < self.max_num_figs and random.random() < self.mut_add_pol:
            cols = map(copy, cols)
            if not copied:
                pols = [map(copy, pol) for pol in pols]
                copied = True
            pol, col = self._generate_figure()
            pols.append(pol)
            cols.append(col)        
            
        return pols, cols
        
    def _one_iteration(self):
        
        # update the best individual ever
        best_index = self.scores.argmax()        
        if self.scores[best_index] > self.best_ind_score:
            self.best_ind_score = self.scores[best_index]
            self.best_ind = self.population[best_index]
        
        children = chain(*[self.population for i in xrange(self.num_children)])
        children = self._mutate_population(children)
        
        P = self.population + children
        scores = np.hstack([self.scores, self._score_population(children)])
        best = scores.argpartition(-self.pop_size)[-self.pop_size:]
        self.population = [P[i] for i in best]
        self.scores = scores[best]
    
    def train(self, num_it, debug=None):
        t0 = time()
        for i in xrange(num_it):
            self._one_iteration()
            self.iterations_done += 1
            
            if debug is not None and not self.iterations_done % debug:
                self.time_elapsed += time() - t0
                t0 = time()
                self.best_imgs.append(self.best_img())
                print 'Score after %i iterations: %i' % (self.iterations_done, self.best_ind_score)
                self.log.append((self.iterations_done, self.best_ind_score, self.best_imgs[-1], self.time_elapsed))
        imshow(self.best_imgs[-1])
        
    def save(self, name):
        name = name + '_hcdyn_%s_%i_%i_%i_%i_%i_%i' % (self.distr, self.pop_size, self.num_children, self.max_num_figs, 
                                                    self.max_num_edges, self.iterations_done, self.seed)
        with open(name + '.pkl', 'w') as f:
            pickle.dump(self.log, f)
            
        self.best_imgs[-1].save(name + '.png', 'png')
                                          
    def best_img(self):
        return draw_individual(self.best_ind, self.ref_img.size, mode=self.mode, board=self.board)