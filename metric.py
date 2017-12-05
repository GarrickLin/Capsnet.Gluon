import mxnet as mx
import numpy as np

class LossMetric(mx.metric.EvalMetric):
    def __init__(self, batch_size, num_gpu):
        super(LossMetric, self).__init__('LossMetric')
        self.batch_size = batch_size
        self.num_gpu = num_gpu
        self.sum_metric = 0
        self.num_inst = 0
        self.loss = 0.0
        self.batch_sum_metric = 0
        self.batch_num_inst = 0
        self.batch_loss = 0.0
        self.recon_loss = 0.0
        self.n_batch = 0

    def update(self, labels, preds):
        batch_sum_metric = 0
        batch_num_inst = 0
        for label, pred_outcaps in zip(labels[0], preds[0]):
            label_np = int(label.asnumpy())
            pred_label = int(np.argmax(pred_outcaps.asnumpy()))
            batch_sum_metric += int(label_np == pred_label)
            batch_num_inst += 1
        batch_loss = preds[1].asnumpy()
        recon_loss = preds[2].asnumpy()
        self.sum_metric += batch_sum_metric
        self.num_inst += batch_num_inst
        self.loss += batch_loss
        self.recon_loss += recon_loss
        self.batch_sum_metric = batch_sum_metric
        self.batch_num_inst = batch_num_inst
        self.batch_loss = batch_loss
        self.n_batch += 1 

    def get_name_value(self):
        acc = float(self.sum_metric)/float(self.num_inst)
        mean_loss = self.loss / float(self.n_batch)
        mean_recon_loss = self.recon_loss / float(self.n_batch)
        return acc, mean_loss, mean_recon_loss

    def get_batch_log(self, n_batch):
        print("n_batch :"+str(n_batch)+" batch_acc:" +
              str(float(self.batch_sum_metric) / float(self.batch_num_inst)) +
              ' batch_loss:' + str(float(self.batch_loss)/float(self.batch_num_inst)))
        self.batch_sum_metric = 0
        self.batch_num_inst = 0
        self.batch_loss = 0.0

    def reset(self):
        self.sum_metric = 0
        self.num_inst = 0
        self.loss = 0.0
        self.recon_loss = 0.0
        self.n_batch = 0