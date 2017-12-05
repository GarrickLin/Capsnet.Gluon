import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from caps_net import CapsNet, margin_loss, mask_mse_loss
import time
from metric import LossMetric
from visdom import Visdom
import numpy as np

viz = Visdom()

class SimpleLRScheduler(mx.lr_scheduler.LRScheduler):
    """A simple lr schedule that simply return `dynamic_lr`. We will set `dynamic_lr`
    dynamically based on performance on the validation set.
    """

    def __init__(self, learning_rate=0.001):
        super(SimpleLRScheduler, self).__init__()
        self.learning_rate = learning_rate

    def __call__(self, num_update):
        return self.learning_rate
    
    
def train_mnist(epochs, input_shape, n_class, num_routing, recon_loss_weight, ctx = mx.gpu(0), log_interval=20, **kwargs):
    batch_size, C, H, W = input_shape
    capsnet = CapsNet(n_class, num_routing, input_shape)
    capsnet.initialize(init=mx.init.Xavier(), ctx=ctx)
    
    capsnet.hybridize()
    
    #mnist = mx.test_utils.get_mnist()
    #train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
    #val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)    
    train_iter = mx.io.MNISTIter(image="data/train-images.idx3-ubyte", 
                                label="data/train-labels.idx1-ubyte",
                                batch_size=batch_size, shuffle=True)
    val_iter = mx.io.MNISTIter(image="data/t10k-images.idx3-ubyte", 
                               label="data/t10k-labels.idx1-ubyte",
                               batch_size=batch_size, shuffle=False)    

    learning_rate = 0.001
    lr_scheduler = SimpleLRScheduler(learning_rate)    
    decay = 0.9
    trainer = gluon.Trainer(capsnet.collect_params(), 
                            optimizer='adam', 
                            optimizer_params = {'lr_scheduler': lr_scheduler})
    

    train_plt = viz.line(Y=np.zeros((1,3)), 
                         X=np.zeros((1,3)), 
                         opts=dict(
                             xlabel='Batch',
                             ylabel='Loss and Acc',
                             title='CapsNet traning plot',
                             legend=['Accuracy', 'Digit Loss', 'Mask Loss']                             
                         ))
    val_plt = viz.line(Y=np.zeros((1,3)), 
                       X=np.zeros((1,3)), 
                       opts=dict(
                           xlabel='Epoch',
                           ylabel='Loss and Acc',
                           title='CapsNet validation plot',
                           legend=['Accuracy', 'Digit Loss', 'Mask Loss']                             
                       ))    
    hist_acc = 0
    #acc_metric = mx.metric.Accuracy()
    loss_metric = LossMetric(batch_size, 1)
    val_metric = LossMetric(batch_size, 1)
    batches_one_epoch = 60000 / batch_size
    for epoch in range(epochs):
        train_iter.reset()
        val_iter.reset()
        loss_metric.reset()        
        for i, batch in enumerate(train_iter):
            tic = time.time()
            x = batch.data[0].as_in_context(ctx)
            y = batch.label[0].as_in_context(ctx)
            y_ori = y
            y = mx.nd.one_hot(y, n_class)
            with autograd.record():
                out_caps, out_mask = capsnet(x, y)
                margin_loss_ = margin_loss(mx.nd, y, out_caps)                
                mask_loss_ = mask_mse_loss(mx.nd, x, out_mask)
                loss = (1-recon_loss_weight)*margin_loss_ + recon_loss_weight*mask_loss_
            loss.backward()
            trainer.step(batch_size)
            loss_metric.update([y_ori], [out_caps, loss, mask_loss_])
            
            if i % log_interval == 0:
                acc, digit_loss, mask_loss = loss_metric.get_name_value()
                viz.line(Y=np.array([acc, digit_loss, mask_loss]).reshape((1,3)),
                         X=np.ones((1,3))*batches_one_epoch*epoch+i,
                         win=train_plt,
                         update='append')
                elasp = time.time()-tic
                print 'Epoch %2d, train %s %.5f, time %.1f sec, %d samples/s' % (epoch, "acc", acc, elasp, int(batch_size/elasp))
        
        lr_scheduler.learning_rate = learning_rate * (decay ** (epoch+1))
        
        val_metric.reset()
        for i, batch in enumerate(val_iter):
            x = batch.data[0].as_in_context(ctx)
            y = batch.label[0].as_in_context(ctx)
            y_ori = y
            y = mx.nd.one_hot(y, n_class)
            out_caps, out_mask = capsnet(x, y)
            margin_loss_ = margin_loss(mx.nd, y, out_caps)                
            mask_loss_ = mask_mse_loss(mx.nd, x, out_mask)
            loss = (1-recon_loss_weight)*margin_loss_ + recon_loss_weight*mask_loss_   
            val_metric.update([y_ori], [out_caps, loss, mask_loss_])
        acc, digit_loss, mask_loss = val_metric.get_name_value()
        viz.line(Y=np.array([acc, digit_loss, mask_loss]).reshape((1,3)),
                 X=np.ones((1,3))*epoch,
                 win=val_plt,
                 update='append')        
        if acc > hist_acc:
            hist_acc = acc
            capsnet.save_params("model/capsnet_%f.params"%acc)
        print 'Epoch %2d, validation %s %.5f' % (epoch, "acc", acc)
                
                                        
if __name__ == "__main__":
    from easydict import EasyDict as edict
    params = edict()
    # epochs, input_shape, n_class, num_routing, recon_loss_weight, ctx = mx.gpu(0)
    params.epochs = 100
    params.batch_size = 80
    params.input_shape = (params.batch_size, 1, 28, 28)
    params.n_class = 10
    params.num_routing = 3
    params.recon_loss_weight = 0.392
    params.log_interval = 1
    params.ctx = mx.gpu(0)
    
    train_mnist(**params)
    
        
        
    