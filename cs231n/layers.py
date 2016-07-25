import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  tempx = x.reshape(x.shape[0],-1)
  out =np.dot(tempx,w)+b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  #all of the bottom follow from taking the derivitive of f = wx +b
  #df/dw = x
  ##df/db = 1
  #df/dx=w
  #then use the chain rule to multiple by the upstream derivative dout
  tempx = (x.reshape(x.shape[0],-1))
  dw = np.dot(tempx.T,dout)
  db = np.sum(dout, axis=0)
  dx = np.dot(dout,w.T).reshape(x.shape)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = x * (x >= 0).astype(float)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  forward = (x>0).astype(float)
  dx = forward*dout
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    #the following comes directly from
    # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html

    #step1: calciulate mean
    mu = (1./N) * np.sum(x, axis = 0)

    #step2: substract mean vector of every trainings example
    xmu = x - mu

    #step3: following the lower branch - calculation denominator
    sq = xmu ** 2

    #step4: calculate variance
    var = (1./N) * np.sum(sq, axis = 0)

    #step5: add eps for numerical stability, then sqrt to get standard dev
    sqrtvar = np.sqrt(var + eps)

    #step6: invert sqrtwar
    ivar = 1./sqrtvar

    #step7: execute normalization
    xhat = xmu * ivar

    #step8: Nor the two transformation steps
    gammax = gamma * xhat

    #step9
    out = gammax + beta

    #store intermediate
    cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)

    running_mean = momentum * running_mean + (1 - momentum) * mu
    running_var = momentum * running_var + (1 - momentum) * var
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################


    #subtract mean and divide by square root of variance (standard deviation)
    out=(x-running_mean)/np.sqrt(running_var + eps)

    #shift by gamma and delta
    out= gamma*out + beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var


  return out, cache

def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.

  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.

  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.

  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  #unfold the variables stored in cache
  xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache

  #get the dimensions of the input/output
  N,D = dout.shape

  #step9
  dbeta = np.sum(dout, axis=0)
  dgammax = dout #not necessary, but more understandable

  #step8
  dgamma = np.sum(dgammax*xhat, axis=0)
  dxhat = dgammax * gamma

  #step7
  divar = np.sum(dxhat*xmu, axis=0)
  dxmu1 = dxhat * ivar

  #step6
  dsqrtvar = -1. /(sqrtvar**2) * divar

  #step5
  dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar

  #step4
  dsq = 1. /N * np.ones((N,D)) * dvar

  #step3
  dxmu2 = 2 * xmu * dsq

  #step2
  dx1 = (dxmu1 + dxmu2)
  dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)

  #step1
  dx2 = 1. /N * np.ones((N,D)) * dmu

  #step0
  dx = dx1 + dx2

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  if mode == 'train':
    mask = (np.random.rand(*x.shape) < p)/p
    out = x*mask
  else:
    mask = None
    out = x

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = dout
  if mode == 'train':
    #WTF changing the following line from dx *=mask to dx = dout*mask fixed error check problem    
    dx =dout*mask
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)  samples,channels, height, width
  - w: Filter weights of shape (F, C, HH, WW)  filters, channels, kernalheight, kernalwidth
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  stride = conv_param['stride']
  pad = conv_param['pad']

  N = x.shape[0]  #number samples
  C = x.shape[1]  #channels
  H = x.shape[2]  #sample height
  W = x.shape[3]  #sample width

  F  = w.shape[0] #number of different filter LAYERS
  HH = w.shape[2] #height of each filter (same for every layer)
  WW = w.shape[3] #width  "

  W_n = int( 1 + (W + 2 * pad - WW) / stride )  #number filters accross
  H_n = int( 1 + (H + 2 * pad - HH) / stride )  #            "  down

  #Pad the input array
  H_with_zero_padding = H + 2*pad
  W_with_zero_padding = W + 2*pad
  XPadded = np.zeros(N*C*H_with_zero_padding*W_with_zero_padding)
  XPadded = XPadded.reshape(N,C,H_with_zero_padding,W_with_zero_padding)
  XPadded[:,:,pad:W+pad,pad:H+pad] = x

  #create our output array (
  out = np.zeros(N*F*H_n*W_n)
  out = out.reshape((N,F,H_n,W_n))

  #now lets iterate over the loop
  for each_sample in range(N):
    for each_layer in range(F):
      for h_n in range(H_n):
        hstart = h_n*stride
        hstop  = hstart + HH

        for w_n in range(W_n):
          wstart = w_n*stride
          wstop = wstart + WW
          out[each_sample,each_layer,h_n,w_n ] = np.sum(XPadded[each_sample,:,hstart:hstop,wstart:wstop] * w[each_layer,:,:,:])+ b[each_layer]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache

def conv_backward_naive(dout, cache):
    """
         A naive implementation of the backward pass for a convolutional layer.

         Inputs:
         - dout: Upstream derivatives.
         - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

         Returns a tuple of:
         - dx: Gradient with respect to x
         - dw: Gradient with respect to w
         - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    # Unwrap cache
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']

    x_pad = np.pad(x, pad_width=[(0,), (0,), (pad,), (pad,)], mode='constant', constant_values=0)

    # Shape the numpy arrays
    dx_pad = np.zeros_like(x_pad) # We will trim off the dx's on the paddings later
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # db calculation
    db = np.sum(dout, axis = (0,2,3))

    # Loading up some values before going to calculate dw and dx
    H_prime = (H+2*pad-HH)/stride+1
    W_prime = (W+2*pad-WW)/stride+1
    #for n in xrange(N):
    for i in xrange(H_prime):
        for j in xrange(W_prime):
            selected_x = x_pad[:,:,i*stride : i*stride+HH, j*stride : j*stride+WW]
            selected_shape = selected_x.shape
            for k in xrange(F):
                dw[k] += np.sum(selected_x*(dout[:,k,i,j])[:,None,None,None], axis=0)
                dx_pad[:,:,i*stride : i*stride+HH, j*stride : j*stride+WW] +=\
                np.einsum('ij,jklm->iklm', dout[:,:,i,j], w)
                dx = dx_pad[:,:,pad:-pad,pad:-pad]
                #############################################################################
                #                             END OF YOUR CODE                              #
                #############################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
      A naive implementation of the forward pass for a max pooling layer.
      Inputs:
      - x: Input data, of shape (N, C, H, W)
      - pool_param: dictionary with the following keys:
        - 'pool_height': The height of each pooling region
        - 'pool_width': The width of each pooling region
        - 'stride': The distance between adjacent pooling regions
      Returns a tuple of:
      - out: Output data
      - cache: (x, pool_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################
    N, C, H, W = x.shape
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']
    H_prime = (H-HH)/stride+1
    W_prime = (W-WW)/stride+1
    out = np.zeros((N,C,H_prime,W_prime))
    
    # This is fairly easy - we just need to compute the shape afterwards : H' and W'
    # And take each pool_height x pool_width square and unwrap it, take the max, and then reshape it.
    for i in xrange(H_prime):
        for j in xrange(W_prime):
            selected_x = x[:,:,i*stride : i*stride+HH, j*stride : j*stride+WW]
            out[:,:,i,j] = np.max(selected_x, axis=(2,3))           
            
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
      A naive implementation of the backward pass for a max pooling layer.
      Inputs:
      - dout: Upstream derivatives
      - cache: A tuple of (x, pool_param) as in the forward pass.
      Returns:
      - dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']
    
    H_prime = (H-HH)/stride+1
    W_prime = (W-WW)/stride+1
    # This is fairly easy - we just need to compute the shape afterwards : H' and W'
    # And take each pool_height x pool_width square and unwrap it, take the max, and then reshape it.
    dx = np.zeros_like(x)
    for i in xrange(H_prime):
        for j in xrange(W_prime):
            selected_x = x[:,:,i*stride : i*stride+HH, j*stride : j*stride+WW]
            val = np.max(selected_x, axis=(2,3)) 
            temp_binary = val[:,:,None,None] == selected_x
            dx[:,:,i*stride : i*stride+HH, j*stride : j*stride+WW] += temp_binary * (dout[:,:,i,j])[:,:,None,None]
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def reshape_to_bn(X, N, C, H, W):
  return np.swapaxes(X, 0, 1).reshape(C, -1).T

def reshape_from_bn(out, N, C, H, W):
  return np.swapaxes(out.T.reshape(C, N, H, W), 0, 1)

#following 2 from https://github.com/OneRaynyDay/CS231n/blob/master/assignment2/cs231n/layers.py
def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
          Computes the forward pass for spatial batch normalization.
          Inputs:
          - x: Input data of shape (N, C, H, W)
          - gamma: Scale parameter, of shape (C,)
          - beta: Shift parameter, of shape (C,)
          - bn_param: Dictionary with the following keys:
            - mode: 'train' or 'test'; required
            - eps: Constant for numeric stability
            - momentum: Constant for running mean / variance. momentum=0 means that
              old information is discarded completely at every time step, while
              momentum=1 means that new information is never incorporated. The
              default of momentum=0.9 should work well in most situations.
            - running_mean: Array of shape (D,) giving running mean of features
            - running_var Array of shape (D,) giving running variance of features
          Returns a tuple of:
          - out: Output data, of shape (N, C, H, W)
          - cache: Values needed for the backward pass
        """
    out, cache = None, None

    #############################################################################
    # Implement the forward pass for spatial batch normalization.               #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################
    N, C, H, W = x.shape
    x_reshaped = x.transpose(0,2,3,1).reshape(N*H*W, C)
    out_tmp, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    out = out_tmp.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
      Computes the backward pass for spatial batch normalization.
      Inputs:
      - dout: Upstream derivatives, of shape (N, C, H, W)
      - cache: Values from the forward pass
      Returns a tuple of:
      - dx: Gradient with respect to inputs, of shape (N, C, H, W)
      - dgamma: Gradient with respect to scale parameter, of shape (C,)
      - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    N, C, H, W = dout.shape
    dout_reshaped = dout.transpose(0,2,3,1).reshape(N*H*W, C)
    dx_tmp, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)
    dx = dx_tmp.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    return dx, dgamma, dbeta
  
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_scores(x):
  """
  Computes the probabilities of a softmax classifier

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.

  Returns:
  - probs: normalized scores
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  return probs


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = softmax_scores(x)

  N = x.shape[0]
  if N>0:
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  else:
    loss = 0
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
