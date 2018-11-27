import os
import argparse
import datetime
import tensorflow as tf

import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer
from utils.pascal_voc import pascal_voc

slim = tf.contrib.slim


class Solver(object):

    #参数：模型net和数据generator
    def __init__(self, net, data):
        self.net = net
        self.data = data
        self.weights_file = cfg.WEIGHTS_FILE
        self.max_iter = cfg.MAX_ITER
        self.initial_learning_rate = cfg.LEARNING_RATE
        self.decay_steps = cfg.DECAY_STEPS
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE
        self.summary_iter = cfg.SUMMARY_ITER            #打印结果间隔，以迭代次数为单位
        self.save_iter = cfg.SAVE_ITER                  #保存模型间隔，以迭代次数为单位
        self.output_dir = os.path.join(                 #该次训练模型输出目录
            cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.save_cfg()                                 #保存每次训练的参数配置，好习惯

        self.variable_to_restore = tf.global_variables() #G返回全局变量列表raphKeys.GLOBAL_VARIABLES
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)  #在checkpoints处保存恢复变量,max_to_keep仅仅保留最近检查点变量
        self.ckpt_file = os.path.join(self.output_dir, 'yolo')                   #检查点文件目录,里面一套文件
        self.summary_op = tf.summary.merge_all()                                 #Merges all summaries collected in the default graph
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)      #记录日志文件Eventfile ，和eager execution不兼容

        self.global_step = tf.train.create_global_step()                         #在graph中创建一个global_step张量
        self.learning_rate = tf.train.exponential_decay(                         #学习率采用指数滑动平均，阶梯状，随着训练步骤指数衰减
            self.initial_learning_rate, self.global_step, self.decay_steps,      #decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
            self.decay_rate, self.staircase, name='learning_rate')
        self.optimizer = tf.train.GradientDescentOptimizer(                      #模型优化器
            learning_rate=self.learning_rate)
        self.train_op = slim.learning.create_train_op(                           #训练张量，约定损失、优化器、全局步
            self.net.total_loss, self.optimizer, global_step=self.global_step)

        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())                        #初始化全局变量

        if self.weights_file is not None:                                       #初始模型权重
            print('Restoring weights from: ' + self.weights_file)
            self.saver.restore(self.sess, self.weights_file)

        self.writer.add_graph(self.sess.graph)                                  #Adds a `Graph` to the event file.为了在tensorboard中显示

    def train(self):

        train_timer = Timer()
        load_timer = Timer()

        for step in range(1, self.max_iter + 1):

            load_timer.tic()
            images, labels = self.data.get()                                        #读取一批数据images(none,448,448,3),labels(none,7,7,25)
            load_timer.toc()
            feed_dict = {self.net.images: images,
                         self.net.labels: labels}

            if step % self.summary_iter == 0:                               #计算并打印训练摘要
                if step % (self.summary_iter * 10) == 0:

                    train_timer.tic()                                       #计算损失值
                    summary_str, loss, _ = self.sess.run(
                        [self.summary_op, self.net.total_loss, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                    log_str = '{} Epoch: {}, Step: {}, Learning rate: {},Loss: {:5.3f}\n' \
                              'Speed: {:.3f}s/iter,Load: {:.3f}s/iter, Remain: {}'.format(
                        datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                        self.data.epoch,                                    #样本完整提取一遍为1轮
                        int(step),
                        round(self.learning_rate.eval(session=self.sess), 6),
                        loss,
                        train_timer.average_time,
                        load_timer.average_time,
                        train_timer.remain(step, self.max_iter))
                    print(log_str)

                else:
                    train_timer.tic()
                    summary_str, _ = self.sess.run(
                        [self.summary_op, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                    # print('{} Epoch:{},Step:{},Speed:{:.3f}s/iter,Remain:{}'.format(
                    #     datetime.datetime.now().strftime('%m-%d %H:%M:%S'), self.data.epoch, int(step),
                    #     train_timer.average_time,train_timer.remain(step, self.max_iter)))

                self.writer.add_summary(summary_str, step)                  #记录日志

            else:                                                           #不打印训练摘要，直接训练
                train_timer.tic()                                           #开始计算时间
                self.sess.run(self.train_op, feed_dict=feed_dict)
                train_timer.toc()                                           #结束本段计算时间，返回平均一批数据的训练时间
                print('{} Epoch:{},Step:{},Speed:{:.3f}s/iter,Remain: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    self.data.epoch,
                    int(step),
                    train_timer.average_time,
                    train_timer.remain(step, self.max_iter)))

            if step % self.save_iter == 0:                                  #保存模型训练参数
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    self.output_dir))
                self.saver.save(
                    self.sess, self.ckpt_file, global_step=self.global_step)

    def save_cfg(self):

        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__                                     #一个模块内的name=value字典项
            for key in sorted(cfg_dict.keys()):                         #按照字典key字典顺序排列
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)


def update_config_paths(data_dir, weights_file):
    cfg.DATA_PATH = data_dir
    cfg.PASCAL_PATH = os.path.join(data_dir, 'pascal_voc')
    cfg.CACHE_PATH = os.path.join(cfg.PASCAL_PATH, 'cache')
    cfg.OUTPUT_DIR = os.path.join(cfg.PASCAL_PATH, 'output')
    cfg.WEIGHTS_DIR = os.path.join(cfg.PASCAL_PATH, 'weights')

    cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--gpu', default='0', type=str)
    args = parser.parse_args()

    if args.gpu is not None:
        cfg.GPU = args.gpu

    if args.data_dir != cfg.DATA_PATH:
        update_config_paths(args.data_dir, args.weights)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    yolo = YOLONet()
    pascal = pascal_voc('train')

    solver = Solver(yolo, pascal)

    print('Start training ...')
    solver.train()
    print('Done training.')


if __name__ == '__main__':

    # python train.py --weights YOLO_small.ckpt --gpu 0
    main()
