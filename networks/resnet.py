import torch.nn as nn
import torchvision.models as models


def get_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(in_features=2048, out_features=1)
    return model
