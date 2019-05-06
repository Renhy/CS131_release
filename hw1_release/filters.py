"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    x_h = int(Hk / 2)
    y_h = int(Wk / 2)
        
    for hi in range(Hi):
        for wi in range(Wi):
            for x in range(Hk):
                for y in range(Wk):
                    v_h = hi - x_h + x
                    v_w = wi - y_h + y
                    if v_h >= 0 and v_h < Hi and v_w >= 0 and v_w < Wi:
                        out[hi, wi] = out[hi, wi] + image[v_h, v_w] * kernel[Hk - x - 1, Wk - y -1]
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.zeros((H + pad_height * 2, W + pad_width * 2))
    for h in range(H):
        for w in range(W):
            out[h + pad_height, w + pad_width] = image[h, w]
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    kernel_flip = np.flip(kernel)
    h_pad = int(Hk / 2)
    w_pad = int(Wk / 2)
    image_pad = zero_pad(image, h_pad, w_pad)
    for h in range(Hi):
        for w in range(Wi):
            out[h, w] = np.sum(image_pad[h:h+Hk, w:w+Wk] * kernel_flip)
    ### END YOUR CODE

    return out


from scipy import signal

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    out = signal.convolve2d(image, kernel)
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g = np.flip(g)
    out = conv_fast(f, g)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g = g - np.mean(g)
    
    g = np.flip(g)
    out = conv_fast(f, g)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    out = np.zeros((Hf, Wf))
    
    h_pad = int(Hg / 2)
    w_pad = int(Wg / 2)
    f_pad = zero_pad(f, h_pad, w_pad)
    
    nor_g = (g - np.mean(g)) / np.std(g)
    for h in range(Hf):
        for w in range(Wf):
            sub_f = f_pad[h:h+Hg, w:w+Wg].copy()
            nor_sub_f = (sub_f - np.mean(sub_f)) / np.std(sub_f)
            
            out[h, w] = np.sum(nor_sub_f * nor_g)
    ### END YOUR CODE

    return out
