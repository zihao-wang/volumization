from .LSTM import LSTMClassifier


def get_model(model, **kwargs):
    if model == "LSTM":
        return LSTMClassifier(**kwargs)
