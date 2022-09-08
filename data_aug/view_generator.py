import numpy as np

np.random.seed(0)


"""1개의 이미지로부터 2개의 Augmented view 생성"""
class ContrastiveLearningViewGenerator(object):

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]