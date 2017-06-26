import numpy as np
from cost import draw_individual_list as draw_individual
from cost import cost
from matplotlib.pyplot import imshow
import random, pickle
from itertools import chain
from copy import deepcopy, copy
from time import time


class HC_Alsing():
    
    def __init__(self, pop_size, num_children, ref_img, board='black',
                 F='default', min_num_edges=3, max_num_edges=10, max_num_figs=50, mode='polygon_simple',
                 mut_add_pol=.002, mut_add_point=.001, mut_rem_point=.001, mut_rem_pol=.001, 
                 mut_alpha=.001, mut_color=.001, mut_pos_hi=.001, mut_pos_med=.001, mut_pos_low=.001, mut_move_pol=.002,
                 lim_alpha_min=0, lim_alpha_max=255, lim_move_pos_min='default', lim_move_pos_max='default', seed=12345):
          
        if F == 'default':
            F = cost
        
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed
            
        if lim_move_pos_min == 'default':
            lim_move_pos_min = (ref_img.size[0] / 50, ref_img.size[1] / 50)
            
        if lim_move_pos_max == 'default':
            lim_move_pos_max = (ref_img.size[0] / 10, ref_img.size[1] / 10)
            
        self.mode = mode
        self.shape = ref_img.size
        
        self.board = board
        
        self.mut_add_pol = mut_add_pol
        self.mut_add_point = mut_add_point
        self.mut_rem_point = mut_rem_point
        self.mut_rem_pol = mut_rem_pol
        self.mut_alpha = mut_alpha
        self.mut_color = mut_color
        self.mut_pos_hi = mut_pos_hi
        self.mut_pos_med = mut_pos_med
        self.mut_pos_low = mut_pos_low
        self.mut_move_pol = mut_move_pol
        
        self.lim_alpha_min = lim_alpha_min
        self.lim_alpha_max = lim_alpha_max
        self.lim_move_pos_min = lim_move_pos_min
        self.lim_move_pos_max = lim_move_pos_max        
        
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
        res = []
        for ind in P:
            mutated = False
            while not mutated:
                new_ind, mutated = self._mutate_individual(ind)
            res.append(new_ind)
        return res
    
    def _mutate_figure(self, pol, col):
        mutated = False
        
        if random.random() < self.mut_rem_point and len(pol) > self.min_num_edges:
            idx = random.randint(0, len(pol) - 1)
            pol = pol[:idx] + pol[idx + 1:]
            mutated = True
            
        if random.random() < self.mut_add_point and len(pol) < self.max_num_edges:
            idx = random.randint(0, len(pol) - 1)
            idx_next = (idx + 1) % len(pol)
            p1 = pol[idx]
            p2 = pol[idx_next]
            new_point = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
            pol = pol[:idx + 1] + [new_point] + pol[idx + 1:]
            mutated = True
            
        mutated_pos_hi = np.where(np.random.rand(len(pol) * 2) < self.mut_pos_hi)[0]
        mutated_pos_med = np.where(np.random.rand(len(pol) * 2) < self.mut_pos_med)[0]
        mutated_pos_low = np.where(np.random.rand(len(pol) * 2) < self.mut_pos_low)[0]
        mutated_col = np.where(np.random.rand(3) < self.mut_color)[0]
        mutated_alpha = random.random() < self.mut_alpha
        
        for i in mutated_pos_hi:
            k, xy = divmod(i, 2)
            vert = copy(pol[k])
            vert[xy] = random.randint(0, self.shape[xy])
            pol = pol[:k] + [vert] + pol[k + 1:]
            mutated = True
        
        for i in mutated_pos_med:
            k, xy = divmod(i, 2)
            vert = copy(pol[k])
            vert[xy] = self._crop(vert[xy] + random.randint(-self.lim_move_pos_max[xy], self.lim_move_pos_max[xy]), xy)
            pol = pol[:k] + [vert] + pol[k + 1:]
            mutated = True
            
        for i in mutated_pos_low:
            k, xy = divmod(i, 2)
            vert = copy(pol[k])
            vert[xy] = self._crop(vert[xy] + random.randint(-self.lim_move_pos_min[xy], self.lim_move_pos_min[xy]), xy)
            pol = pol[:k] + [vert] + pol[k + 1:]
            mutated = True
            
        if mutated_col.size or mutated_alpha:
            col = copy(col)
            mutated = True
            
            for i in mutated_col:
                col[i] = random.randint(0, 255)
            
            if mutated_alpha:
                col[-1] = random.randint(self.lim_alpha_min, self.lim_alpha_max)
        
        return pol, col, mutated
    
    def _mutate_individual(self, ind):
        pols, cols = ind
        mutated = False
        
        if pols:
            if random.random() < self.mut_rem_pol:
                idx = random.randint(0, len(pols) - 1)
                pols = pols[:idx] + pols[idx + 1:]
                mutated = True

            if random.random() < self.mut_move_pol:
                idx1 = random.randint(0, len(pols) - 1)
                idx2 = random.randint(0, len(pols) - 1)
                pols, cols = copy(pols), copy(cols)               
                pol, col = pols.pop(idx1), cols.pop(idx2)
                pols.insert(idx2, pol)
                cols.insert(idx2, col)
                mutated = True

        if random.random() < self.mut_add_pol and len(pols) < self.max_num_figs:
            idx = random.randint(0, len(pols))
            pol, col = self._generate_figure()
            pols = pols[:idx] + [pol] + pols[idx:]
            cols = cols[:idx] + [col] + cols[idx:]
            mutated = True
            
        res = ([], [])
        
        for i in xrange(len(pols)):
            new_pol, new_col, mutated_fig = self._mutate_figure(pols[i], cols[i])
            mutated |= mutated_fig
            res[0].append(new_pol)
            res[1].append(new_col)
            
        return res, mutated
        
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
        name = name + '_hcAls_%i_%i_%i_%i_%i_%i' % (self.pop_size, self.num_children, self.max_num_figs, 
                                                   self.max_num_edges, self.iterations_done, self.seed)
        with open(name + '.pkl', 'w') as f:
            pickle.dump(self.log, f)
            
        self.best_imgs[-1].save(name + '.png', 'png')
                                          
    def best_img(self):
        return draw_individual(self.best_ind, self.ref_img.size, mode=self.mode, board=self.board)
    
##########################################################
# Robocza wersja HC_Alsing, na ktorej testowalem rozne zmiany.
    
class HC_Alsing_mod():
    
    def __init__(self, pop_size, num_children, ref_img, board='black',
                 F='default', min_num_edges=3, max_num_edges=10, max_num_figs=50, mode='polygon_simple',
                 mut_add_pol=.002, mut_add_point=.001, mut_rem_point=.001, mut_rem_pol=.001, 
                 mut_alpha=.001, mut_color=.001, mut_pos_hi=.001, mut_pos_med=.001, mut_pos_low=.001, mut_move_pol=.002,
                 lim_alpha_min=30, lim_alpha_max=60, lim_move_pos_min='default', lim_move_pos_max='default', seed=12345):
          
        if F == 'default':
            F = cost
        
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed
            
        if lim_move_pos_min == 'default':
            lim_move_pos_min = (ref_img.size[0] / 50, ref_img.size[1] / 50)
            
        if lim_move_pos_max == 'default':
            lim_move_pos_max = (ref_img.size[0] / 10, ref_img.size[1] / 10)
            
        self.mode = mode
        self.shape = ref_img.size
        
        self.board = board
        
        self.mut_add_pol = mut_add_pol
        self.mut_add_point = mut_add_point
        self.mut_rem_point = mut_rem_point
        self.mut_rem_pol = mut_rem_pol
        self.mut_alpha = mut_alpha
        self.mut_color = mut_color
        self.mut_pos_hi = mut_pos_hi
        self.mut_pos_med = mut_pos_med
        self.mut_pos_low = mut_pos_low
        self.mut_move_pol = mut_move_pol
        
        self.lim_alpha_min = lim_alpha_min
        self.lim_alpha_max = lim_alpha_max
        self.lim_move_pos_min = lim_move_pos_min
        self.lim_move_pos_max = lim_move_pos_max        
        
        if self.mode == 'polygon_simple':
            self.max_num_edges = max_num_edges
            self.min_num_edges = min_num_edges
        elif self.mode == 'circle':
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
        
        for i in xrange(self.pop_size):
            for j in xrange(self.max_num_figs):
                p = [[random.randint(0, self.shape[0]), random.randint(0, self.shape[1])] for k in xrange(3)]
                c = [random.randint(0, 255) for k in xrange(4)]
                self.population[i][0].append(p)
                self.population[i][1].append(c)
        
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
    
    def _mutate_figure(self, pol, col):
        if random.random() < self.mut_rem_point and len(pol) > self.min_num_edges:
            idx = random.randint(0, len(pol) - 1)
            pol = pol[:idx] + pol[idx + 1:]
            
        if random.random() < self.mut_add_point and len(pol) < self.max_num_edges:
            idx = random.randint(0, len(pol) - 1)
            idx_next = (idx + 1) % len(pol)
            p1 = pol[idx]
            p2 = pol[idx_next]
            new_point = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
            pol = pol[:idx + 1] + [new_point] + pol[idx + 1:]
            
        mutated_pos_hi = np.where(np.random.rand(len(pol) * 2) < self.mut_pos_hi)[0]
        mutated_pos_med = np.where(np.random.rand(len(pol) * 2) < self.mut_pos_med)[0]
        mutated_pos_low = np.where(np.random.rand(len(pol) * 2) < self.mut_pos_low)[0]
        mutated_col = np.where(np.random.rand(3) < self.mut_color)[0]
        mutated_alpha = random.random() < self.mut_alpha
        
        for i in mutated_pos_hi:
            k, xy = divmod(i, 2)
            vert = copy(pol[k])
            vert[xy] = int(random.triangular(0, self.shape[xy]))
            pol = pol[:k] + [vert] + pol[k + 1:]
        
        for i in mutated_pos_med:
            k, xy = divmod(i, 2)
            vert = copy(pol[k])
            vert[xy] = self._crop(vert[xy] + random.randint(-self.lim_move_pos_max[xy], self.lim_move_pos_max[xy]), xy)
            pol = pol[:k] + [vert] + pol[k + 1:]
            
        for i in mutated_pos_low:
            k, xy = divmod(i, 2)
            vert = copy(pol[k])
            vert[xy] = self._crop(vert[xy] + random.randint(-self.lim_move_pos_min[xy], self.lim_move_pos_min[xy]), xy)

            pol = pol[:k] + [vert] + pol[k + 1:]
            
        if mutated_col.size or mutated_alpha:
            col = copy(col)
            
            for i in mutated_col:
                col[i] = random.randint(0, 255)
            
            if mutated_alpha:
                col[-1] = random.randint(self.lim_alpha_min, self.lim_alpha_max)
        
        return pol, col
    
    def _mutate_individual(self, ind):
        pols, cols = ind
        
        if pols:
            if random.random() < self.mut_rem_pol:
                idx = random.randint(0, len(pols) - 1)
                pols = pols[:idx] + pols[idx + 1:]

            if random.random() < self.mut_move_pol:
                idx1 = random.randint(0, len(pols) - 1)
                idx2 = random.randint(0, len(pols) - 1)
                pols, cols = copy(pols), copy(cols)                
                pol, col = pols.pop(idx1), cols.pop(idx2)
                pols.insert(idx2, pol)
                cols.insert(idx2, col)

        if random.random() < self.mut_add_pol and len(pols) < self.max_num_figs:
            idx = random.randint(0, len(pols))
            pol, col = self._generate_figure()
            pols = pols[:idx] + [pol] + pols[idx:]
            cols = cols[:idx] + [col] + cols[idx:]
            
        res = ([], [])
        
        for i in xrange(len(pols)):
            new_pol, new_col = self._mutate_figure(pols[i], cols[i])
            res[0].append(new_pol)
            res[1].append(new_col)
            
        return res
        
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
        name = name + '_hcAlsmod_%i_%i_%i_%i_%i_%i' % (self.pop_size, self.num_children, self.max_num_figs, 
                                                   self.max_num_edges, self.iterations_done, self.seed)
        with open(name + '.pkl', 'w') as f:
            pickle.dump(self.log, f)
            
        self.best_imgs[-1].save(name + '.png', 'png')
                                          
    def best_img(self):
        return draw_individual(self.best_ind, self.ref_img.size, mode=self.mode, board=self.board)