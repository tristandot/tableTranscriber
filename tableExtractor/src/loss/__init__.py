from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


def get_loss(name):
    if name is None:
        name = 'bce'
    return {
        'bce': BCEWithLogitsLoss,
        'cross_entropy': CrossEntropyLoss,
        'mean_squared_error': MSELoss,
    }[name]
