
import torch
import torch.nn as nn
from model_dhp.resnet_dhp_share import BottleNeck_dhp, ResBlock_dhp
from model_dhp.dhp_base import DHP_Base, conv_dhp

resnet18_config = [2, 2, 2, 2]
resnet34_config = [3, 4, 6, 3]
resnet50_config = [3, 4, 6, 3]
resnet101_config = [3, 4, 23, 3]
resnet152_config = [3, 8, 36, 3]


def make_model(args, parent=False):
    return ResNet_ImageNet_DHP(args)


class ResNet_ImageNet_DHP(DHP_Base):
    def __init__(self, args):
        super(ResNet_ImageNet_DHP, self).__init__(args=args)

        self.width_mult = args.width_mult
        if args.depth >= 50:
            self.expansion = 4
            self.block = BottleNeck_dhp
        else:
            self.expansion = 1
            self.block = ResBlock_dhp
        self.linear_percentage = args.linear_percentage

        self.config = eval('resnet{}_config'.format(args.depth))
        depth = 0
        self.depth_cum = []
        for d in self.config:
            depth += d
            self.depth_cum.append(depth)

        self.in_channels = int(64 * self.width_mult)

        self.latent_vector_stage0 = nn.Parameter(torch.randn((3)))
        self.latent_vector_stage1 = nn.Parameter(torch.randn(int(64 * self.expansion * self.width_mult)))
        self.latent_vector_stage2 = nn.Parameter(torch.randn(int(128 * self.expansion * self.width_mult)))
        self.latent_vector_stage3 = nn.Parameter(torch.randn(int(256 * self.expansion * self.width_mult)))
        self.latent_vector_stage4 = nn.Parameter(torch.randn(int(512 * self.expansion * self.width_mult)))

        stride = 2
        v = self.latent_vector_stage1 if self.expansion == 1 else None

        self.features = nn.ModuleList([conv_dhp(args.n_colors, int(64 * self.width_mult), kernel_size=7, stride=stride, latent_vector=v, embedding_dim=self.embedding_dim),
                                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)])
        self.features.extend(self.make_layer(self.config[0], int(64 * self.width_mult), self.kernel_size, latent_vector=self.latent_vector_stage1))
        self.features.extend(self.make_layer(self.config[1], int(128 * self.width_mult), self.kernel_size, stride=2, latent_vector=self.latent_vector_stage2))
        self.features.extend(self.make_layer(self.config[2], int(256 * self.width_mult), self.kernel_size, stride=2, latent_vector=self.latent_vector_stage3))
        self.features.extend(self.make_layer(self.config[3], int(512 * self.width_mult), self.kernel_size, stride=2, latent_vector=self.latent_vector_stage4))
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(int(512 * self.expansion * self.width_mult), self.n_classes)
        self.show_latent_vector()

    def make_layer(self, blocks, planes, kernel_size, stride=1, conv3x3=conv_dhp, latent_vector=None):
        out_channels = planes * self.expansion
        if stride != 1 or self.in_channels != out_channels:
            downsample = conv3x3(self.in_channels, out_channels, 1, stride=stride, act=False, latent_vector=latent_vector, embedding_dim=self.embedding_dim)
        else:
            downsample = None
        kwargs = {'conv3x3': conv3x3, 'embedding_dim': self.embedding_dim}
        m = [self.block(self.in_channels, planes, kernel_size, stride=stride, downsample=downsample, latent_vector=latent_vector, **kwargs)]
        self.in_channels = out_channels
        for _ in range(blocks - 1):
            m.append(self.block(self.in_channels, planes, kernel_size, latent_vector=latent_vector, **kwargs))

        return m

    def mask(self):
        masks = []
        for i, v in enumerate(self.gather_latent_vector()):
            if i == 4:
                channels = self.remain_channels(v, percentage=self.linear_percentage)
            else:
                channels = self.remain_channels(v)
            if i == 0:
                masks.append(torch.ones_like(v, dtype=torch.bool, device='cuda'))
            else:
                masks.append(v.abs() >= min(self.pt, v.abs().topk(channels)[0][-1]))
        # mask has no gradients. Comparison operations are without gradients automatically.
        # when calculating the weights, biases, don't need to slice them?
        return masks

    def proximal_operator(self, lr):
        regularization = self.regularization * lr
        for i, v in enumerate(self.gather_latent_vector()):
            if i == 4:
                channels = self.remain_channels(v, percentage=self.linear_percentage)
            else:
                channels = self.remain_channels(v)
            if i != 0:
                if torch.sum(v.abs() >= self.pt) > channels:
                    self.soft_thresholding(v, regularization)

    def convert_index(self, i):
        if i == 0:
            x = 0
        elif i - 1 == 1:
            if self.expansion == 1:
                x = 1
            else:
                x = 5
        elif 1 < i - 1 <= self.depth_cum[0] + 1:
            x = 1
        elif self.depth_cum[0] + 1 < i - 1 <= self.depth_cum[1] + 1:
            x = 2
        elif self.depth_cum[1] + 1 < i - 1 <= self.depth_cum[2] + 1:
            x = 3
        else:
            x = 4
        return x

    def set_parameters(self, calc_weight=False):
        latent_vectors, masks = self.gather_latent_vector(), self.mask()
        for i, layer in enumerate(self.features):
            j = self.convert_index(i)
            if self.expansion == 1:
                if i == 0:
                    vm = [latent_vectors[j]] + masks[:j + 2]
                elif i in [d + 2 for d in self.depth_cum[:3]]:
                    vm = [latent_vectors[j], masks[j], masks[i + 3], masks[j + 1]]
                else:
                    vm = [latent_vectors[j], masks[j], masks[i + 3]]
            elif self.expansion == 4:
                if i == 0:
                    vm = [latent_vectors[j], masks[j], masks[5]]
                else:
                    mask = masks[(i-1) * 2 + 4: (i-1) * 2 + 6]
                    if i == 1:
                        pass
                    elif i == 2:
                        vm = [latent_vectors[j], masks[j]] + mask + [masks[1]]
                    elif i in [d + 2 for d in self.depth_cum[:3]]:
                        vm = [latent_vectors[j], masks[j]] + mask + [masks[j + 1]]
                    else:
                        vm = [latent_vectors[j], masks[j]] + mask
            else:
                raise NotImplementedError('Expansion type {} not implemented for ResNet'.format(self.expansion))
            if i != 1:
                layer.set_parameters(vm, calc_weight)
        mask_input = masks[4].to(torch.float32).nonzero().squeeze(1)
        if calc_weight:
            weight = self.classifier.weight.data
            weight = torch.index_select(weight, dim=1, index=mask_input)
            self.classifier.weight = nn.Parameter(weight)
            self.classifier.in_features = mask_input.shape[0]
        self.classifier.in_features_remain = mask_input.shape[0]

    def forward(self, x):
        if not self.finetuning:
            latent_vectors = self.gather_latent_vector()
            for i, layer in enumerate(self.features):
                j = self.convert_index(i)
                if i == 1:
                    x = layer(x)
                else:
                    x = layer([x, latent_vectors[j]])
        else:
            for i, layer in enumerate(self.features):
                x = layer(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


