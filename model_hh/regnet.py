from model.regnet import RegNet

def make_model(args, parent=False):
    return RegNet(args)
