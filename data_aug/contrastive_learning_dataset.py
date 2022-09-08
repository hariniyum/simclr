from torchvision.transforms import transforms
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    """Data augmentation transformations 적용"""
    # @staticmethod: 
    # staticmethod 아래 정의된 함수를 클래스에 대한 객체 작업을 거치지 않고 클래스 외부에서 사용할 수 있게 해줌
    @staticmethod
    def get_simclr_pipeline_transform(size, s=0.5):
        # s: Color distortion의 Strength로 CIFAR-10의 경우 0.5 사용
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        gaussian_blur = transforms.GaussianBlur(kernel_size=int(0.1*size))
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(), 
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              # transforms.RandomApply(gaussian_blur), # CIFAR-10에서는 사용하지 않음
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                self.get_simclr_pipeline_transform(32),
                                                                n_views),
                                                                download=True),
                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                            self.get_simclr_pipeline_transform(96),
                                                            n_views),
                                                            download=True)}
        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
