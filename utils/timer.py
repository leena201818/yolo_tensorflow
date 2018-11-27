import time
import datetime


class Timer(object):
    '''
    A simple timer.
    '''

    def __init__(self):
        self.init_time = time.time()
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.remain_time = 0.

    #计时器开始
    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    #计算一次时间，返回平均每次调用的时间，或者计时器启动以来的时间
    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    #估计剩余时间——根据总迭代次数和当前迭代次数
    def remain(self, iters, max_iters):
        if iters == 0:
            self.remain_time = 0
        else:
            self.remain_time = (time.time() - self.init_time) * \
                (max_iters - iters) / iters
        return str(datetime.timedelta(seconds=int(self.remain_time)))
