import numpy as np

import torchvision.datasets as datasets


class CIFAR10to5(datasets.CIFAR10):
    """CIFAR10to5 Dataset.

    Args:
        seed (int): Random seed (default: 12345).
        
    This is a subclass of the `CIFAR10` Dataset.
    """

    def __init__(self, seed=12345, **kwargs):
        super(CIFAR10to5, self).__init__(**kwargs)
        self.seed = seed
        np.random.seed(self.seed)
        targets = np.array(self.targets)
        for i, target in enumerate(targets):
            if np.random.uniform(0, 1) <= 0.5:
                targets[i] = (target // 2) % 5
            else:
                targets[i] = (target // 2 + target % 2) % 5
        targets = [int(x) for x in targets]
        self.targets = targets


class CIFAR7to3(datasets.CIFAR10):
    """CIFAR7to3 Dataset.

    Args:
        seed (int): Random seed (default: 12345).
        
    This is a subclass of the `CIFAR10` Dataset.
    """

    def __init__(self, seed=12345, **kwargs):
        super(CIFAR7to3, self).__init__(**kwargs)
        self.seed = seed
        np.random.seed(self.seed)
        targets = np.array(self.targets)
        self.data = self.data[targets <= 6]
        targets = targets[targets <= 6]
        for i, target in enumerate(targets):
            if target < 6:
                rnd = np.random.uniform(0, 1)
                if rnd <= 0.5:
                    targets[i] = (target // 2) % 3
                else:
                    targets[i] = (target // 2 + target % 2) % 3
            else:
                rnd = np.random.uniform(0, 1)
                if rnd <= 1 / 3.:
                    targets[i] = 0
                elif rnd <= 2 / 3.:
                    targets[i] = 1
                else:
                    targets[i] = 2
        targets = [int(x) for x in targets]
        self.targets = targets
