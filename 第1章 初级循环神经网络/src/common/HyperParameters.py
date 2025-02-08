# 超级参数
class HyperParameters(object):
    def __init__(self, max_epoch=1000, batch_size=32, learning_rate=0.01):
        self.max_epoch = max_epoch              # 最大的训练轮数
        self.batch_size = batch_size        # 批大小
        self.learning_rate = learning_rate  # 学习率（初始的，后期可能会变化）
