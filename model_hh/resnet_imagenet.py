from torchvision import models

def make_model(args, parent=False):
    net = 'resnet' + str(args.depth)
    net = getattr(models, net)(pretrained=False)
    return net