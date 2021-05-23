from model_dhp.regnet_imagenet_dhp import RegNet_DHP
# from model_hh import output_feature_map_hook

def make_model(args, parent=False):
    return RegNet_DHP(args)
