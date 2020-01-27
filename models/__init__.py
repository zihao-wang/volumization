from .LSTM import LSTMClassifier
from .resnet import ResNet18
from .DNN import DNN
from .LSTM_Attn import AttentionModel
from .efficientnet_pytorch import EfficientNet


def get_model(model, **kwargs):
    if model == "LSTM":
        return LSTMClassifier(**kwargs)
    elif model == "LSTMATT":
        return AttentionModel(**kwargs)
    elif model == "ResNet18":
        return ResNet18(**kwargs)
    elif model == "DNN":
        return DNN()
    elif model == "EfficientNet":
        return EfficientNet.from_name('efficientnet-b1')
    else:
        raise IOError("Model not defined")
