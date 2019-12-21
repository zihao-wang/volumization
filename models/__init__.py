from .LSTM import LSTMClassifier
from .resnet import ResNet18

def get_model(model, **kwargs):
    if model == "LSTM":
        return LSTMClassifier(**kwargs)
    elif model == "ResNet18":
        return ResNet18()
    else:
        raise IOError("Model not defined")
