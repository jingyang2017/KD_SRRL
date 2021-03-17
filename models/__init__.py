from .official_resnet import resnet34T,resnet18S,resnet50T
from .MobileNet import MobileNet

model_dict = {
    'resnet50T':resnet50T,
    'resnet34T':resnet34T,
    'resnet18S':resnet18S,
    'MobileNet':MobileNet
}
