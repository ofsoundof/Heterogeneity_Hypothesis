import torch
import torch.nn as nn
from model.regnet import cfg_x_200mf, cfg_x_400mf, cfg_x_600mf, cfg_x_800mf, \
    cfg_y_200mf, cfg_y_400mf, cfg_y_600mf, cfg_y_800mf
from model_dhp.efficientnet_dhp import conv_bn_act
from model_dhp.dhp_base import DHP_Base, addindent
from IPython import embed



def make_model(args):
    return RegNet_DHP(args)


class SE(nn.Module):
    """Squeeze-and-Excitation block with Swish."""

    def __init__(self, in_channel, se_channels):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channel, se_channels, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(se_channels, in_channel, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def set_parameters(self, mask, calc_weight=False):
        in_channels = sum(mask[0].cpu().to(torch.int).numpy())
        se_channels = sum(mask[1].cpu().to(torch.int).numpy())
        self.fc[0].in_channels_remain = in_channels
        self.fc[0].out_channels_remain = se_channels
        self.fc[1].channels = se_channels
        self.fc[2].in_channels_remain = se_channels
        self.fc[2].out_channels_remain = in_channels
        self.fc[3].channels = in_channels
        if calc_weight:
            mask0 = mask[0].to(torch.float32).nonzero().squeeze(1)
            mask1 = mask[1].to(torch.float32).nonzero().squeeze(1)
            weight1, bias1 = self.fc[0].weight, self.fc[0].bias
            # embed()
            weight1 = torch.index_select(torch.index_select(weight1, dim=0, index=mask1), dim=1, index=mask0)
            bias1 = torch.index_select(bias1, dim=0, index=mask1)
            self.fc[0].weight = nn.Parameter(weight1)
            self.fc[0].bias = nn.Parameter(bias1)
            self.fc[0].in_channels = in_channels
            self.fc[0].out_channels = se_channels

            weight2, bias2 = self.fc[2].weight, self.fc[2].bias
            weight2 = torch.index_select(torch.index_select(weight2, dim=0, index=mask0), dim=1, index=mask1)
            bias2 = torch.index_select(bias2, dim=0, index=mask0)
            self.fc[2].weight = nn.Parameter(weight2)
            self.fc[2].bias = nn.Parameter(bias2)
            self.fc[2].in_channels = se_channels
            self.fc[2].out_channels = in_channels

    def forward(self, x):
        out = self.fc(self.avg_pool(x))
        out = x * out
        return out


class ResBasicBlock(nn.Module):
    """Residual basic block: x + F(x), F = basic transform"""
    def __init__(self, w_in, w_out, stride, group_width, bottleneck_ratio, reduction, embedding_dim=8, latent_vector=None):
        super(ResBasicBlock, self).__init__()
        self.finetuning = False
        w_b = int(round(w_out * bottleneck_ratio))
        num_groups = w_b // group_width
        self.group_width = group_width
        self.latent_vector = nn.Parameter(torch.randn(num_groups))
        # 1x1
        self.layer1 = conv_bn_act(w_in, w_b, 1, 1, act='relu', embedding_dim=embedding_dim, latent_vector=self.latent_vector)
        # 3x3
        self.layer2 = conv_bn_act(w_b, w_b, 3, stride, groups=num_groups, act='relu', embedding_dim=embedding_dim)
        # se
        with_se = reduction > 1
        if with_se:
            self.se = SE(w_b, w_in // reduction)
        # 1x1
        self.layer3 = conv_bn_act(w_b, w_out, 1, 1, embedding_dim=embedding_dim, latent_vector=latent_vector)

        self.skip_proj = None
        if stride != 1 or w_in != w_out:
            self.skip_proj = conv_bn_act(w_in, w_out, 1, stride, embedding_dim=embedding_dim, latent_vector=latent_vector)
        self.relu = nn.ReLU(inplace=True)

    def set_parameters(self, vector_mask, calc_weight=False):
        # if self.finetuning:
        # embed()
        self.layer1.set_parameters(vector_mask[:3], calc_weight)
        self.layer2.set_parameters([self.latent_vector] + vector_mask[2:4], calc_weight)
        if hasattr(self, 'se'):
            self.se.set_parameters([vector_mask[2].repeat_interleave(self.group_width), vector_mask[1]], calc_weight)
        if self.skip_proj is not None:
            self.layer3.set_parameters([self.latent_vector] + vector_mask[2::2], calc_weight)
            self.skip_proj.set_parameters(vector_mask[:2] + [vector_mask[-1]], calc_weight)
        else:
            self.layer3.set_parameters([self.latent_vector] + vector_mask[2:0:-1], calc_weight) #TODO: make sure this is correct
        self.relu.channels = self.layer3.out_channels_remain if hasattr(self.layer3, 'out_channels_remain') \
            else self.layer3.out_channels

    def forward(self, x):
        if not self.finetuning:
            x, latent_vector_input = x
            out = self.layer1([x, latent_vector_input])
            out = self.layer2([out, self.latent_vector])
            if hasattr(self, 'se'):
                out = self.se(out)
            out = self.layer3([out, self.latent_vector])
            if self.skip_proj is not None:
                x = self.skip_proj([x, latent_vector_input])
        else:
            out = self.layer2(self.layer1(x))
            if hasattr(self, 'se'):
                out = self.se(out)
            out = self.layer3(out)
            if self.skip_proj is not None:
                x = self.skip_proj(x)
        out += x
        out = self.relu(out)
        return out


class RegNet_DHP(DHP_Base):
    def __init__(self, args):
        super(RegNet_DHP, self).__init__(args)
        self.cfg = eval('cfg_' + args.regime)
        self.linear_percentage = args.linear_percentage
        self.width_mult = args.width_mult
        self.data_train = args.data_train
        depth = 0
        self.depth_cum = []
        for d in self.cfg['depths']:
            depth += d
            self.depth_cum.append(depth)
        self.group_width = self.cfg['group_width']
        print(self.depth_cum)
        if args.data_train.find('CIFAR') >= 0:
            num_classes = int(args.data_train[5:])
        elif args.data_train.find('Tiny') >= 0:
            num_classes = 200
        else:
            num_classes = 1000

        stride = 2 if args.data_train == 'ImageNet' else 1
        self.in_planes = int(32 * self.width_mult)
        self.latent_vectors = nn.ParameterList([
            nn.Parameter(torch.randn(3)),
            nn.Parameter(torch.randn(self.in_planes // 4))
        ])
        self.features = nn.ModuleList([conv_bn_act(3, self.in_planes, 3, stride, act='relu', embedding_dim=self.embedding_dim, latent_vector=self.latent_vectors[-1])])
        self.features.extend(self._make_layer(0))
        self.features.extend(self._make_layer(1))
        self.features.extend(self._make_layer(2))
        self.features.extend(self._make_layer(3))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(int(self.width_mult * self.cfg['widths'][-1]), num_classes)
        self.show_latent_vector()

    def _make_layer(self, idx):
        depth = self.cfg['depths'][idx]
        width = int(self.width_mult * self.cfg['widths'][idx])
        stride = self.cfg['strides'][idx]
        group_width = self.cfg['group_width']
        bottleneck_ratio = self.cfg['bottleneck_ratio']
        reduction = self.cfg['se_reduction']
        self.latent_vectors.append(nn.Parameter(torch.randn(width // 4)))
        layers = []
        for i in range(depth):
            if self.data_train.find('CIFAR') >= 0 and idx in [0, 1]:
                s = 1
            else:
                s = stride if i == 0 else 1
            # s = stride if i == 0 else 1
            layers.append(ResBasicBlock(self.in_planes, width, s, group_width, bottleneck_ratio, reduction, embedding_dim=self.embedding_dim, latent_vector=self.latent_vectors[-1]))
            self.in_planes = width
        return layers

    def convert_index(self, i):
        if i == 0:
            x = 0
        elif i == 1:
            x = 1
        elif 1 < i <= self.depth_cum[0] + 1:
            x = 2
        elif self.depth_cum[0] + 1 < i <= self.depth_cum[1] + 1:
            x = 3
        elif self.depth_cum[1] + 1 < i <= self.depth_cum[2] + 1:
            x = 4
        else:
            x = 5
        return x

    def mask(self):
        masks = []
        for i, v in enumerate(self.gather_latent_vector(grad_prune=self.grad_prune, grad_normalize=self.grad_normalize)):
            if i == 5:
                channels = self.remain_channels(v, percentage=self.linear_percentage)
            else:
                channels = self.remain_channels(v)
            if i != 0 and v.shape[0] != self.group_width:
                masks.append(v.abs() >= min(self.pt, v.abs().topk(channels)[0][-1]))
            else:
                masks.append(torch.ones_like(v, dtype=torch.bool, device='cuda'))

        # mask has no gradients. Comparison operations are without gradients automatically.
        # when calculating the weights, biases, don't need to slice them?
        return masks

    def proximal_operator(self, lr):
        regularization = self.regularization * lr
        for i, v in enumerate(self.gather_latent_vector()):
            if i == 5:
                channels = self.remain_channels(v, percentage=self.linear_percentage)
            else:
                channels = self.remain_channels(v)
            if i != 0 and v.shape[0] != self.group_width:
                vector = v if not self.grad_prune else v.grad
                if torch.sum(vector.abs() >= self.pt) > channels:
                    self.soft_thresholding(v, regularization)

    def set_parameters(self, calc_weight=False):
        latent_vectors, masks = self.gather_latent_vector(), self.mask()
        # former_vectors, last_vectors, other_vectors, former_masks, last_masks, other_masks = self.mask()
        for i, layer in enumerate(self.features):
            j = self.convert_index(i)
            # print(i)
            # if self.finetuning:
            #     embed()
            if i == 0:
                vm = [latent_vectors[j], masks[j], masks[j + 1]]

            elif i in [1] + [1 + d for d in self.depth_cum[:3]]:
                # if self.finetuning:
                #     embed()
                vm = [latent_vectors[j], masks[j]] + masks[2 * i + 4: 2 * i + 6] + [masks[j + 1]]
            else:
                vm = [latent_vectors[j], masks[j]] + masks[2 * i + 4: 2 * i + 6]
            s = [v.shape for v in vm]
            # print(layer)
            # print(s)
            layer.set_parameters(vm, calc_weight)
        # embed()
        mask = masks[5]
        mask_input = mask.repeat_interleave(4).to(torch.float32).nonzero().squeeze(1)
        self.classifier.in_features_remain = mask_input.shape[0]
        if calc_weight:
            self.classifier.in_features = mask_input.shape[0]
            weight = self.classifier.weight.data
            weight = torch.index_select(weight, dim=1, index=mask_input)
            self.classifier.weight = nn.Parameter(weight)

    def forward(self, x):
        if not self.finetuning:
            latent_vectors = self.gather_latent_vector()
            for i, layer in enumerate(self.features):
                # print(i)
                j = self.convert_index(i)
                x = layer([x, latent_vectors[j]])
        else:
            for i, layer in enumerate(self.features):
                x = layer(x)
        out = self.avg_pool(x)
        out = out.view(out.size()[0], -1)
        out = self.classifier(out)
        # embed()
        return out