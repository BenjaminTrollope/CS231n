import numpy as np

# X is a single column vector
# y is an integer specifying the label
# W is the weight matrix

def L_i_vectorized(x, y, W):
    scores = W.dot(x)   # evaluate scores, w*x
    margins = np.maximum(0, scores - scores[y] + 1)  # margins = difference scores that is obtained and  the correct score +1
    margins[y] = 0 # effeciently takes away a 1 that would inflate score
    loss_i = np.sum(margins)
    return loss_i
