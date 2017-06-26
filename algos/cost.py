import numpy as np
from PIL import Image, ImageDraw
import skimage.color as color


def draw_individual_array(ind, shape, mode='polygon_simple', board='black'):

    X, Y = shape
    img = Image.new('RGB', shape, board)
    drw = ImageDraw.Draw(img, 'RGBA')    
    
    num_figs, pol_len = ind.shape
    position_len = pol_len - 4
    colors = map(tuple, (ind[:, position_len:pol_len] * 255).astype(np.int32))
    
    if mode == 'polygon_simple':
        assert not position_len % 2
        vertices = ind[:, :position_len].reshape(num_figs, -1, 2) * np.array([[[X, Y]]])

        for i in xrange(num_figs):
            polygon = map(tuple, vertices[i])
            drw.polygon(polygon, colors[i])
    elif mode == 'circle':
        assert position_len == 3
        positions = ind[:, :position_len - 1].reshape(num_figs, 2) * np.array([X, Y])
        radii = ind[:, position_len - 1] / np.sqrt(2) * np.sqrt(X**2 + Y**2)

        for i in xrange(num_figs):
            middle = positions[i]
            t = radii[i]
            circle = (middle[0] - t, middle[1] - t, middle[0] + t, middle[1] + t)
            drw.ellipse(circle, colors[i])
        
    return img

def draw_individual_list(ind, shape, mode='polygon_simple', board='black'):

    X, Y = shape
    img = Image.new('RGB', shape, board)
    drw = ImageDraw.Draw(img, 'RGBA')
    
    figs, colors = ind[:2]
    figs = [map(tuple, fig) for fig in figs]
    colors = map(tuple, colors)
    
    if mode == 'polygon_simple':
        for pol, col in zip(figs, colors):
            drw.polygon(pol, col)
            
    elif mode == 'circle':
        raise NotImplementedError
        
    return img


def dist(img, matrix, mode='euclid'):
    img = np.array(img, dtype=np.int32)
    
    if mode == 'euclid':
        return ((img - matrix)**2).sum(axis=2).mean() # mean squared per-pixel distance
    else:
        raise Exception('%s is not a valid metric' % mode)
        
        
def cost(ind, ref_img, dist_mode='euclid', mode='polygon_simple', ref_matrix=None, board='white'):
    if isinstance(ind, np.ndarray):
        img = draw_individual_array(ind, ref_img.size, mode=mode, board=board)
    else:
        img = draw_individual_list(ind, ref_img.size, mode=mode, board=board)
    if ref_matrix is None:
        ref_matrix = np.array(ref_img, dtype=np.int32)
    return int(dist(img, ref_matrix, mode=dist_mode))


def cost_lab(ind, ref_img, dist_mode='euclid', mode='polygon_simple', ref_matrix=None):
    if isinstance(ind, np.ndarray):
        img = draw_individual_array(ind, ref_img.size, mode=mode)
    else:
        img = draw_individual_list(ind, ref_img.size, mode=mode)
    return int(dist_lab(img, ref_img))

    
def dist_lab(img, ref):
    return (color.deltaE_ciede2000(color.rgb2lab(ref), color.rgb2lab(img))**2).sum()