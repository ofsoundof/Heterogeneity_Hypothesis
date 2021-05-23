
import torch
import torch.nn as nn
from model_dhp.dhp_base import DHP_Base, conv_dhp
from IPython import embed


def make_model(args, parent=False):
    return ResNet_DHP(args)


class ResBlock_dhp(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, stride=1, conv3x3=conv_dhp, downsample=None, embedding_dim=8):
        super(ResBlock_dhp, self).__init__()
        expansion = 1
        self.finetuning = False
        self.stride = stride
        self.layer1 = conv3x3(in_channels, planes, kernel_size, stride=stride, embedding_dim=embedding_dim)
        self.layer2 = conv3x3(planes, expansion * planes, kernel_size, act=False, embedding_dim=embedding_dim)
        self.downsample = downsample

        # if stride != 1 or in_channels != planes:
        #     self.downsample = conv3x3(in_channels, planes * expansion, 1, stride=stride, act=False, embedding_dim=embedding)
        self.act_out = nn.ReLU()

    def set_parameters(self, vector_mask, calc_weight=False):
        self.layer1.set_parameters(vector_mask[:3], calc_weight)
        self.layer2.set_parameters([self.layer1.latent_vector] + vector_mask[2:4], calc_weight)
        if self.downsample is not None:
            self.downsample.set_parameters(vector_mask[:2] + [vector_mask[-1]], calc_weight)

    def forward(self, x):
        if not self.finetuning:
            x, latent_input_vector = x
            out = self.layer1([x, latent_input_vector])

            out = self.layer2([out, self.layer1.latent_vector])
            if self.downsample is not None:
                x = self.downsample([x, latent_input_vector])
        else:
            out = self.layer2(self.layer1(x))
            if self.downsample is not None:
                x = self.downsample(x)
        out += x
        out = self.act_out(out)
        return out


class BottleNeck_dhp(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, stride=1, conv3x3=conv_dhp, downsample=None, embedding_dim=8):
        super(BottleNeck_dhp, self).__init__()
        expansion = 4
        self.finetuning = False
        self.stride = stride
        self.layer1 = conv3x3(in_channels, planes, 1, embedding_dim=embedding_dim)
        self.layer2 = conv3x3(planes, planes, kernel_size, stride=stride, embedding_dim=embedding_dim)
        self.layer3 = conv3x3(planes, expansion*planes, 1, act=False, embedding_dim=embedding_dim)
        self.downsample = downsample

        # if stride != 1 or in_channels != planes:
        #     self.downsample = conv3x3(in_channels, planes * expansion, 1, stride=stride, act=False, embedding_dim=embedding)
        self.act_out = nn.ReLU()

    def set_parameters(self, vector_mask, calc_weight=False):
        self.layer1.set_parameters(vector_mask[:3], calc_weight)
        self.layer2.set_parameters([self.layer1.latent_vector] + vector_mask[2:4], calc_weight)
        self.layer3.set_parameters([self.layer2.latent_vector] + vector_mask[3:5], calc_weight)
        if self.downsample is not None:
            self.downsample.set_parameters(vector_mask[:2] + [vector_mask[-1]], calc_weight)

    def forward(self, x):
        if not self.finetuning:
            x, latent_input_vector = x
            out = self.layer1([x, latent_input_vector])
            out = self.layer2([out, self.layer1.latent_vector])
            out = self.layer3([out, self.layer2.latent_vector])
            if self.downsample is not None:
                x = self.downsample([x, latent_input_vector])
        else:
            out = self.layer3(self.layer2(self.layer1(x)))
            if self.downsample is not None:
                x = self.downsample(x)
        out += x
        out = self.act_out(out)

        return out


class ResNet_DHP(DHP_Base):
    def __init__(self, args):
        super(ResNet_DHP, self).__init__(args=args)

        if args.depth <= 56:
            self.expansion = 1
            self.block = ResBlock_dhp
            self.n_blocks = (args.depth - 2) // 6
        else:
            self.expansion = 4
            self.block = BottleNeck_dhp
            self.n_blocks = (args.depth - 2) // 9
        self.in_channels = 16
        self.downsample_type = 'C'
        self.prune_same_channels = args.prune_same_channels == 'Yes'

        self.latent_vector = nn.Parameter(torch.randn((3)))
        stride = 1 if args.data_train.find('CIFAR') >= 0 else 2
        self.features = nn.ModuleList([conv_dhp(args.n_colors, 16, kernel_size=self.kernel_size, stride=stride, embedding_dim=self.embedding_dim)])
        self.features.extend(self.make_layer(self.n_blocks, 16, self.kernel_size))
        self.features.extend(self.make_layer(self.n_blocks, 32, self.kernel_size, stride=2))
        self.features.extend(self.make_layer(self.n_blocks, 64, self.kernel_size, stride=2))
        self.pooling = nn.AvgPool2d(8)
        self.classifier = nn.Linear(64 * self.expansion, self.n_classes)

    def make_layer(self, blocks, planes, kernel_size, stride=1, conv3x3=conv_dhp):
        out_channels = planes * self.expansion
        if stride != 1 or self.in_channels != out_channels:
            downsample = conv3x3(self.in_channels, out_channels, 1, stride=stride, act=False, embedding_dim=self.embedding_dim)
        else:
            downsample = None
        kwargs = {'conv3x3': conv3x3, 'embedding_dim': self.embedding_dim}
        m = [self.block(self.in_channels, planes, kernel_size, stride=stride, downsample=downsample, **kwargs)]
        self.in_channels = out_channels
        for _ in range(blocks - 1):
            m.append(self.block(self.in_channels, planes, kernel_size, **kwargs))

        return m

    def gather_latent_vector(self):
        """
        :return: former_vectors -> the latent vectors of the former layers in the Bottleneck or ResBlock
                 last_vectors -> the latent vectors of the last layer in the Bottleneck or ResBlock
                 other_vectors -> the other latent vectors in ResNet including, the 1) input and 2) output latent vector
                 of the first conv, the 3) & 4) latent vectors of the the downsampling block.
        """
        flag = 2 if self.expansion == 1 else 3
        former_vectors, last_vectors, other_vectors = [], [], []
        for k, v in self.state_dict(keep_vars=True).items():
            if k.find('latent_vector') >= 0:
                if k.find('layer' + str(flag)) >= 0:
                    last_vectors.append(v)
                elif k == 'latent_vector' or k == 'features.0.latent_vector' or k.find('downsample') >= 0:
                    other_vectors.append(v)
                else:
                    former_vectors.append(v)
        return former_vectors, last_vectors, other_vectors

    def mask(self):
        former_vectors, last_vectors, other_vectors = self.gather_latent_vector()
        # mask has no gradients. Comparison operations are without gradients automatically.
        # when calculating the weights, biases, don't need to slice them?
        former_masks = [vector.abs() >= min(self.pt, vector.abs().topk(self.mc)[0][-1]) for vector in former_vectors]
        last_masks = []
        other_masks = [torch.ones(3, dtype=torch.bool, device='cuda'),
                       torch.ones_like(other_vectors[-1], dtype=torch.bool, device='cuda')]  # input of first conv and output of last downsample
        for i in range(2):
            vectors = last_vectors[i * self.n_blocks: (i + 1) * self.n_blocks] + [other_vectors[i + 1]]
            if not self.prune_same_channels:
                n_elements = min([torch.sum(v.abs() >= min(self.pt, v.abs().topk(self.mc)[0][-1])) for v in vectors])
                last_masks.extend(v.abs() >= v.abs().topk(n_elements)[0][-1] for v in vectors)
                other_masks.insert(i + 1, last_masks.pop())
            else:
                vector_norm = torch.stack(vectors, dim=1).norm(p=2, dim=1) / vectors[0].shape[0]
                mask = vector_norm >= min(self.pt, vector_norm.topk(self.mc)[0][-1])
                last_masks.extend([mask] * self.n_blocks)
                other_masks.insert(i + 1, mask)
        last_masks.extend(torch.ones_like(v, dtype=torch.bool, device='cuda') for v in last_vectors[2 * self.n_blocks:])
        return former_vectors, last_vectors, other_vectors, former_masks, last_masks, other_masks

    def soft_thresholding_group(self, latent_matrix, reg):
        eps = 1e-8
        vector_norm = torch.norm(latent_matrix, p=2, dim=1) / latent_matrix.shape[1]
        # if torch.isnan(n[0]):
        #     embed()
        scale = torch.max(1 - reg / (vector_norm + eps), torch.zeros_like(vector_norm, device=vector_norm.device))
        return scale

    def proximal_operator(self, lr):
        regularization = self.regularization * lr
        former_vectors, last_vectors, other_vectors = self.gather_latent_vector()
        last_vectors = last_vectors[:2 * self.n_blocks]
        other_vectors = other_vectors[1:]
        if not self.prune_same_channels:
            for v in former_vectors + last_vectors:
                if torch.sum(v.abs() >= self.pt) > self.mc:
                    self.soft_thresholding(v, regularization)
        else:
            for v in former_vectors:
                if torch.sum(v.abs() >= self.pt) > self.mc:
                    self.soft_thresholding(v, regularization)
            for i in range(2):
                vectors = last_vectors[i * self.n_blocks: (i + 1) * self.n_blocks] + [other_vectors[i]]
                latent_matrix = torch.stack(vectors, dim=1)
                vector_norm = latent_matrix.norm(p=2, dim=1) / latent_matrix.shape[1]
                # print(vector_norm)
                if torch.sum(vector_norm >= self.pt) > self.mc:
                    scale = self.soft_thresholding_group(latent_matrix, regularization)
                    for v in vectors:
                        v.data = v.data * scale

    def set_parameters(self, calc_weight=False):
        former_vectors, last_vectors, other_vectors, former_masks, last_masks, other_masks = self.mask()
        offset = 1 if self.expansion == 1 else 2
        for i, layer in enumerate(self.features):
            if i == 0:
                vm = [other_vectors[0]] + other_masks[:2]
            else:
                mask = former_masks[(i - 1) * offset: i * offset] if self.expansion == 4 else [former_masks[i - 1]]
                if i == 1:
                    vm = [other_vectors[1], other_masks[1]] + mask + [last_masks[0]]
                elif (i - 1) % self.n_blocks == 0 and i > 1:
                    vm = [last_vectors[i - 2], last_masks[i - 2]] + mask + [last_masks[i - 1], other_masks[(i - 1) // self.n_blocks + 1]]
                else:
                    vm = [last_vectors[i - 2], last_masks[i - 2]] + mask + [last_masks[i - 1]]
            layer.set_parameters(vm, calc_weight)

    def forward(self, x):
        if not self.finetuning:
            former_vectors, last_vectors, other_vectors = self.gather_latent_vector()
            for i, layer in enumerate(self.features):
                if i <= 1:
                    x = layer([x, other_vectors[i]])
                else:
                    x = layer([x, last_vectors[i-2]])
        else:
            for i, layer in enumerate(self.features):
                x = layer(x)
        x = self.pooling(x)
        x = self.classifier(x.squeeze())
        return x


# class conv_dhp(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=False, batchnorm=True, act=True, args=None):
#         """
#         :param in_channels:
#         :param out_channels:
#         :param kernel_size:
#         :param stride:
#         :param bias:
#         :param batchnorm: whether to append Batchnorm after the activations.
#         :param act: whether to append ReLU after the activations.
#         :param args:
#         """
#         super(conv_dhp, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.batchnorm = batchnorm
#         self.act = act
#         self.finetuning = False
#
#         # hypernetwork
#         latent_dim = args.latent_dim
#         bound = math.sqrt(3) * math.sqrt(1 / 1)
#         weight1 = torch.randn((in_channels, out_channels, latent_dim)).uniform_(-bound, bound)
#         # Hyperfan-in
#         bound = math.sqrt(3) * math.sqrt(1 / (latent_dim * in_channels * kernel_size ** 2))
#         weight2 = torch.randn((in_channels, out_channels, kernel_size ** 2, latent_dim)).uniform_(-bound, bound)
#
#         self.latent_vector = nn.Parameter(torch.randn((out_channels)))
#         self.weight1 = nn.Parameter(weight1)
#         self.weight2 = nn.Parameter(weight2)
#         self.bias0 = nn.Parameter(torch.zeros(in_channels, out_channels))
#         self.bias1 = nn.Parameter(torch.zeros(in_channels, out_channels, latent_dim))
#         self.bias2 = nn.Parameter(torch.zeros(in_channels, out_channels, kernel_size ** 2))
#         # self.bn_hyper = nn.BatchNorm2d(in_channels)
#
#         # main network
#         self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
#         if self.batchnorm:
#             self.bn_main = nn.BatchNorm2d(out_channels)
#         if self.act:
#             self.relu = nn.ReLU()
#
#     def __repr__(self):
#         if hasattr(self, 'in_channels_remain') and hasattr(self, 'out_channels_remain'):
#             s = '\n(1): Conv_dhp({}, {}, kernel_size=({}, {}), stride={})'.format(
#                 self.in_channels_remain, self.out_channels_remain, self.kernel_size, self.kernel_size, self.stride)
#         else:
#             s = '\n(1): Conv_dhp({}, {}, kernel_size=({}, {}), stride={})'.format(
#                 self.in_channels, self.out_channels, self.kernel_size, self.kernel_size, self.stride)
#         if self.batchnorm:
#             s = s + '\n(2): ' + repr(self.bn_main)
#         if self.act:
#             s = s + '\n(3): ' + repr(self.relu)
#         s = addindent(s, 2)
#         return s
#
#     def set_parameters(self, vector_mask, calc_weight=False):
#         latent_vector_input, mask_input, mask_output = vector_mask
#         mask_input = mask_input.to(torch.float32).nonzero().squeeze()
#         mask_output = mask_output.to(torch.float32).nonzero().squeeze()
#         if calc_weight:
#             # embed()
#             latent_vector_input = torch.index_select(latent_vector_input, dim=0, index=mask_input)
#             latent_vector_output = torch.index_select(self.latent_vector, dim=0, index=mask_output)
#             weight = self.calc_weight(latent_vector_input, latent_vector_output, mask_input, mask_output)
#
#             bias = self.bias if self.bias is None else torch.index_select(self.bias, dim=0, index=mask_output)
#             bn_weight = torch.index_select(self.bn_main.weight, dim=0, index=mask_output)
#             bn_bias = torch.index_select(self.bn_main.bias, dim=0, index=mask_output)
#             bn_mean = torch.index_select(self.bn_main.running_mean, dim=0, index=mask_output)
#             bn_var = torch.index_select(self.bn_main.running_var, dim=0, index=mask_output)
#
#             self.weight = nn.Parameter(weight)
#             self.bias = bias if bias is None else nn.Parameter(bias)
#             self.bn_main.weight = nn.Parameter(bn_weight)
#             self.bn_main.bias = nn.Parameter(bn_bias)
#             self.bn_main.running_mean = bn_mean
#             self.bn_main.running_var = bn_var
#         self.in_channels_remain = mask_input.shape[0]
#         self.out_channels_remain = mask_output.shape[0]
#         self.bn_main.num_features = mask_output.shape[0]
#         self.bn_main.num_features_remain = mask_output.shape[0]
#
#     def calc_weight(self, latent_vector_input, latent_vector_output=None, mask_input=None, mask_output=None):
#         if latent_vector_output is None:
#             latent_vector_output = self.latent_vector
#         if mask_input is None and mask_output is None:
#             bias0, bias1, bias2, weight1, weight2 = self.bias0, self.bias1, self.bias2, self.weight1, self.weight2
#         else:
#             bias0 = torch.index_select(torch.index_select(self.bias0, dim=0, index=mask_input), dim=1,
#                                        index=mask_output)
#             bias1 = torch.index_select(torch.index_select(self.bias1, dim=0, index=mask_input), dim=1,
#                                        index=mask_output)
#             bias2 = torch.index_select(torch.index_select(self.bias2, dim=0, index=mask_input), dim=1,
#                                        index=mask_output)
#             weight1 = torch.index_select(torch.index_select(self.weight1, dim=0, index=mask_input), dim=1,
#                                          index=mask_output)
#             weight2 = torch.index_select(torch.index_select(self.weight2, dim=0, index=mask_input), dim=1,
#                                          index=mask_output)
#         # embed()
#         weight = torch.matmul(latent_vector_input.unsqueeze(-1), latent_vector_output.unsqueeze(0)) + bias0
#         weight = weight.unsqueeze(-1) * weight1 + bias1
#         weight = torch.matmul(weight2, weight.unsqueeze(-1)).squeeze(-1) + bias2
#         # if weight.nelement() != self.in_channels * self.out_channels * self.kernel_size ** 2:
#         #     embed()
#         in_channels = latent_vector_input.nelement()
#         out_channels = latent_vector_output.nelement()
#         weight = weight.reshape(in_channels, out_channels, self.kernel_size, self.kernel_size).permute(1, 0, 2, 3)
#         # weight = self.bn_hyper(weight)
#         return weight
#
#     def forward(self, input):
#         if not self.finetuning:
#             out = self.forward_pruning(input)
#         else:
#             out = self.forward_finetuning(input)
#         if self.batchnorm:
#             out = self.bn_main(out)
#         if self.act:
#             out = self.relu(out)
#         return out
#
#     def forward_pruning(self, x):
#         x, latent_vector_input = x
#
#         # execute the hypernetworks to get the weights of the backbone network
#         weight = self.calc_weight(latent_vector_input)
#
#         out = F.conv2d(x, weight, bias=self.bias, stride=self.stride, padding=self.kernel_size // 2)
#         return out
#
#     def forward_finetuning(self, input):
#         out = F.conv2d(input, self.weight, bias=self.bias, stride=self.stride, padding=self.kernel_size // 2)
#         return out
