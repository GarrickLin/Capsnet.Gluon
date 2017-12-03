import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from caps_net import CapsNet, margin_loss, mask_mse_loss
import time

def train_mnist(epochs, input_shape, n_class, num_routing, recon_loss_weight, ctx = mx.gpu(0), log_interval=20, **kwargs):
    batch_size, C, H, W = input_shape
    capsnet = CapsNet(n_class, num_routing, input_shape)
    capsnet.initialize(init=mx.init.Xavier(), ctx=ctx)
    
    #capsnet.hybridize()
    
    mnist = mx.test_utils.get_mnist()
    train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)    

    
    trainer = gluon.Trainer(capsnet.collect_params(), 
                            'adam', {'learning_rate': 0.0005, 'wd': 5e-4})
    

    acc_metric = mx.metric.Accuracy()
    for epoch in range(epochs):
        train_iter.reset()
        val_iter.reset()
        acc_metric.reset()        
        for i, batch in enumerate(train_iter):
            tic = time.time()
            x = batch.data[0].as_in_context(ctx)
            y = batch.label[0].as_in_context(ctx)
            y = mx.nd.one_hot(y, n_class)
            with autograd.record():
                out_caps, out_mask = capsnet(x, y)
                loss1 = margin_loss(mx.nd, y, out_caps)
                loss2 = mask_mse_loss(mx.nd, x, out_mask)
                loss = (1-recon_loss_weight)*loss1 + recon_loss_weight*loss2
            loss.backward()
            trainer.step(batch_size)
            acc_metric.update(y, out_caps)
            
            if i % log_interval == 0:
                acc_str, acc_val = acc_metric.get()
                elasp = time.time()-tic
                print 'Epoch %2d, train %s %.5f, time %.1f sec, %f samples/s' % (epoch, acc_str, acc_val, elasp, batch_size/elasp)
        
                
                                        
if __name__ == "__main__":
    from easydict import EasyDict as edict
    params = edict()
    # epochs, input_shape, n_class, num_routing, recon_loss_weight, ctx = mx.gpu(0)
    params.epochs = 100
    params.batch_size = 10
    params.input_shape = (params.batch_size, 1, 28, 28)
    params.n_class = 10
    params.num_routing = 3
    params.recon_loss_weight = 0.392
    params.log_interval = 1
    params.ctx = mx.gpu(0)
    
    train_mnist(**params)
    
        
        
    