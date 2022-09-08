import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, dataset="cifar10"):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {'resnet18': models.resnet18(pretrained=False, num_classes=out_dim),
                            'resnet50': models.resnet50(pretrained=False, num_classes=out_dim)}

        self.dataset = dataset
        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # 2-layer Projection head 추가
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            # 데이터셋이 CIFAR-10이라면 구조 변경
            if self.dataset == "cifar10": 
                model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
                model.maxpool = nn.Identity()

        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet 18 or resnet50"
            )
        else:
            return model

    def forward(self, x):
        return self.backbone(x)