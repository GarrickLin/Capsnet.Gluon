from mxnet import gluon
from mxnet.gluon import nn
import numpy as np 

eps = np.finfo(float).eps

def squash(F, vectors, axis=-1):
    s_squared_norm = F.sum(F.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / ((1+s_squared_norm) * F.sqrt(s_squared_norm+eps))
    return F.broadcast_mul(scale, vectors)


def length(F, inputs):
    return F.sqrt(F.sum(F.square(inputs), -1))


def Mask(F, inputs):
    if isinstance(inputs, list):
        assert len(inputs) == 2
        inputs, mask = inputs
        mask = F.expand_dims(mask, -1)
    else:
        # compute lengths of capsules
        x = F.sqrt(F.sum(F.square(inputs), -1, True))
        # Enlarge the range of values in x to make max(new_x[i,:])=1 and others << 0
        x = (x - F.max(x, 1, True)) / eps + 1
        # the max value in x clipped to 1 and other to 0. Now `mask` is one-hot coding.
        mask = F.clip(x, 0, 1)        
    
    return F.elemwise_mul(inputs, mask)
    
    
class PrimaryCap(gluon.HybridBlock):
    def __init__(self, dim_capsule, n_channels, kernel_size, strides, padding, **kwargs):
        super(PrimaryCap, self).__init__(**kwargs)
        self.dim_capsule = dim_capsule
        with self.name_scope():
            self.conv2vec = nn.Conv2D(channels=dim_capsule*n_channels, kernel_size=kernel_size, 
                                      strides=strides, padding=padding)
            
    def hybrid_forward(self, F, x):
        vecs = self.conv2vec(x)
        vecs = F.reshape(vecs, shape=(0, -1, self.dim_capsule))
        vecs = squash(F, vecs)
        return vecs
    

#class CapsuleLayer(gluon.HybridBlock):
    #def __init__(self, num_capsule, dim_capsule, num_routing=3, 
                 #in_shape=None, weight_initializer=None, 
                 #bias_initializer='zeros',  **kwargs):       
        #super(CapsuleLayer, self).__init__(**kwargs)
        #with self.name_scope():
            #self.num_routing = num_routing
            #assert len(in_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
            #self.input_num_capsule = in_shape[-2]
            #self.input_dim_capsule = in_shape[-1]
            #self.batch_size = in_shape[0]
            #self.num_capsule = num_capsule
            #self.dim_capsule = dim_capsule            
            #self.weight = self.params.get('weight', 
                                          #shape=(num_capsule*self.input_num_capsule, self.input_dim_capsule, dim_capsule),
                                          #init=weight_initializer, allow_deferred_init=True)
            ##self.bias = self.params.get('bias', shape=(),
                                        ##init=bias_initializer,
                                        ##allow_deferred_init=True)
            
    #def hybrid_forward(self, F, x, weight):
        #print x.shape
        ## expand input dims
        ## x.shape = [None, input_num_capsule, input_dim_capsule]
        ## x_expand.shape = [None, 1, input_num_capsule, input_dim_capsule]
        #x_expand = F.expand_dims(x, 1)
        ## Replicate num_capsule dimension to prepare being multiplied by W
        ## x_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule]
        #x_tiled = F.tile(x_expand, [1, self.num_capsule, 1, 1])        
        ## x_reshape.shape [batch_size*num_capsule*input_num_capsule, 1, input_dim_capsule]
        #x_reshape = F.reshape(x_tiled, (-1, 1, self.input_dim_capsule))
        ## weight_rep.shape [batch_size*num_capsule*input_num_capsule, input_dim_capsule, dim_capsule]
        #weight_rep = F.repeat(weight, self.batch_size, axis=0)
        
        ## x_hat_sqeeze.shape [batch_size*num_capsule*input_num_capsule, 1, dim_capsule]
        #x_hat_sqeeze = F.batch_dot(x_reshape, weight_rep)
        ## x_hat.shape [batch_size*num_capsule, input_num_capsule, dim_capsule]
        #x_hat = F.reshape(x_hat_sqeeze, (-1, self.input_num_capsule, self.dim_capsule))
        
        ## Routing algorithm
        ## In forward pass, `inputs_hat_stopped` = `inputs_hat`;
        ## In backward, no gradient can flow from `inputs_hat_stopped` back to `inputs_hat`.        
        #x_hat_stopped = F.stop_gradient(x_hat)
        
        ## The prior for coupling coefficient, initialized as zeros.
        #b = F.zeros(shape=[self.batch_size, self.num_capsule, self.input_num_capsule])
        
        #assert self.num_routing > 0, 'The num_routing should be > 0.'
        #for i in range(self.num_routing):
            #c = F.softmax(b, axis=1)
            #c = F.reshape(c, (-1, 1, self.input_num_capsule))
            ## At last iteration, use `inputs_hat` to compute `outputs` in order to backpropagate gradient
            #if i == self.num_routing - 1:
                ## c.shape [batch_size*num_capsule, 1, input_num_capsule]
                ## x_hat.shape [batch_size*num_capsule, input_num_capsule, dim_capsule]
                ## outputs.shape [batch_size*num_capsule, 1, dim_capsule]
                #outputs = squash(F.batch_dot(c, x_hat))  # [None, 10, 16]  
            #else:
                ## c.shape [batch_size*num_capsule, 1, input_num_capsule]
                ## x_hat_stopped.shape [batch_size*num_capsule, input_num_capsule, dim_capsule]     
                ## outputs.shape = [batch_size*num_capsule, 1, dim_capsule]
                #outputs = squash(K.batch_dot(c, x_hat_stopped))
                ## x_hat_stopped_trans.shape [batch_size*num_capsule, dim_capsule, input_num_capsule]
                #x_hat_stopped_trans = F.transpose(x_hat_stopped, (0,2,1))
                ## b_out.shape [batch_size*num_capsule, 1, input_num_capsule]
                #b_out = F.batch_dot(outputs, x_hat_stopped)   
                ## b_out.shape [batch_size, num_capsule, input_num_capsule]
                #b_out = F.reshape(b_out, (self.batch_size, self.num_capsule, self.input_num_capsule))
                #b = F.elemwise_add(b, b_out)
                
        #return F.reshape(outputs, (batch_size, num_capsule, dim_capsule))
        

def matmul(F, m1, m2, axis=1):
    #print "m1", m1.shape
    #print "m2", m2.shape
    bmul = F.broadcast_mul(m1, m2)
    #print "bmul", bmul.shape
    bsum = F.sum(bmul, axis=axis, keepdims=True)                
    #print "bsum", bsum.shape     
    return bsum
    
        
class CapsuleLayer(gluon.HybridBlock):
    def __init__(self, num_capsule, dim_capsule, num_routing=3, 
                 in_shape=None, weight_initializer=None, 
                 bias_initializer='zeros',  **kwargs):       
        super(CapsuleLayer, self).__init__(**kwargs)
        with self.name_scope():
            self.num_routing = num_routing
            assert in_shape is not None, "in_shape should not be none"
            assert len(in_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
            self.input_num_capsule = in_shape[-2]
            self.input_dim_capsule = in_shape[-1]
            self.batch_size = in_shape[0]
            self.num_capsule = num_capsule
            self.dim_capsule = dim_capsule            
            self.weight = self.params.get('weight', 
                                          shape=(1, self.input_num_capsule, num_capsule, self.input_dim_capsule, dim_capsule),
                                          init=weight_initializer, allow_deferred_init=True)
    
    def hybrid_forward(self, F, x, weight):
        # x : (batch_size, input_num_capsule, input_dim_vector)
        # inputs_expand : (batch_size, input_num_capsule, 1, input_dim_vector, 1)
        #inputs_expand = F.expand_dims(x, 3)
        #print "inputs_expand 1", inputs_expand.shape
        #inputs_expand = F.expand_dims(inputs_expand, 2)
        #print "inputs_expand 2", inputs_expand.shape
        
        #print "x", x.shape
        inputs_expand = F.reshape(x, (0, 0, -4, -1, 1))
        #print "inputs_expand 1", inputs_expand.shape
        inputs_expand = F.reshape(inputs_expand, (0, 0, -4, 1, -1, 0))
        #print "inputs_expand 2", inputs_expand.shape
        
        # input_tiled (batch_size, input_num_capsule, num_capsule, input_dim_vector, 1)
        inputs_tiled = F.tile(inputs_expand, (1, 1, self.num_capsule, 1, 1))
        #print "inputs_tiled", inputs_tiled.shape
        # w_tiled : (batch_size, input_num_capsule, num_capsule, input_dim_vector, dim_vector)
        w_tiled = F.tile(weight, (self.batch_size, 1, 1, 1, 1))
        #print "w_tiled", w_tiled.shape
        
        # inputs_hat : (1L, input_num_capsule, num_capsule, 1, dim_vector)
        inputs_hat  = F.linalg_gemm2(w_tiled, inputs_tiled, transpose_a=True)
        #print "inputs_hat", inputs_hat.shape
        
        inputs_hat = F.swapaxes(data=inputs_hat, dim1=3, dim2=4)
        inputs_hat_stopped = F.stop_gradient(inputs_hat)
        
        bias = F.zeros((self.batch_size, self.input_num_capsule, self.num_capsule, 1, 1))
        bias_ = F.stop_gradient(bias)
        
        for i in range(self.num_routing):
            c = F.softmax(bias_, axis=2)
            #print "c", c.shape
            if i == self.num_routing - 1:
                #print "i == self.num_routing - 1"
                bsum = matmul(F, c, inputs_hat)
                outputs = squash(F, bsum)
                #print "outputs", outputs.shape
            else:
                #print "i < self.num_routing - 1"
                bsum = matmul(F, c, inputs_hat_stopped)
                outputs = squash(F, bsum)
                #print "outputs", outputs.shape    
                bias_ = bias_ + matmul(F, c, inputs_hat_stopped, axis=4)
                #print "bias_", bias_.shape
                
        outputs = F.reshape(outputs, (-1, self.num_capsule, self.dim_capsule))
        return outputs
    

                
                
                
        
        
        
        
        