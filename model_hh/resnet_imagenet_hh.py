from model_dhp.resnet_imagenet_dhp import ResNet_ImageNet_DHP
# from model_hh import output_feature_map_hook

def make_model(args, parent=False):
    return ResNet_ImageNet_DHP(args)
