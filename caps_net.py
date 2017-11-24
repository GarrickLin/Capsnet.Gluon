import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from caps_layers import PrimaryCap, CapsuleLayer, length, Mask, squash

class CapsNet(gluon.HybridBlock):
    def __init__(self, input_shape, n_class, num_routing, **kwargs):
        super(CapsNet, self).__init__(**kwargs)
        N, C, W, H = input_shape
        with self.name_scope():
            self.net = nn.HybridSequential(prefix='')
            self.net.add(nn.Conv2D(256, kernel_size=9, strides=1, padding=0, 
                                   activation='relu'))
            self.net.add(PrimaryCap(dim_capsule=8, n_channels=32, kernel_size=9, strides=2, 
                                    padding=0))
            self.net.add(CapsuleLayer(num_capsule=n_class, dim_capsule=16, in_shape=input_shape))
            
            self.decoder = nn.HybridSequential(prefix='')
            self.decoder.add(nn.Dense(512, activation='relu'))
            self.decoder.add(nn.Dense(1024, activation='relu'))
            self.decoder.add(nn.Dense(W*H, activation='sigmoid'))
            
    def hybrid_forward(self, F, x, y):
        digitcaps = self.net(x)
        out_caps = length(F, digitcaps)
        # decode network        
        masked_by_y = Mask(F, [digitcaps, y])
        out_mask = self.decoder(masked_by_y)
        out_mask = F.reshape(out_mask, (N,C,W,H))
        
        return out_caps, out_mask
    
def margin_loss(F, y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """    
    L = y_true * F.square(F.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * F.square(F.maximum(0., y_pred - 0.1))
    
        
        