from .LSTM import LSTMClassifier
from .resnet import ResNet18
from .DNN import DNN
from .LSTM_Attn import AttentionModel


def get_model(model, **kwargs):
    if model == "LSTM":
        return LSTMClassifier(**kwargs)
    elif model == "LSTMATT":
        return AttentionModel(**kwargs)
    elif model == "ResNet18":
        return ResNet18()
    elif model == "DNN":
        return DNN()
    else:
        raise IOError("Model not defined")
