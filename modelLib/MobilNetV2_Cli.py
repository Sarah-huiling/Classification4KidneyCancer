from torch import nn
import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torchvision
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, in_c=3, num_classes=2, info_num=7, width_mult=1.0, inverted_residual_setting=None, round_nearest=2):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        self.drop_path_prob = 0.0

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(in_c, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(nn.Dropout(0.2),)
        self.info_conv1 = nn.Linear(info_num, 128)  # 7个临床特征，上采样为5
        self.new_fc = nn.Linear(self.last_channel + 128, num_classes)

        # self.sigmoid = nn.Sigmoid()  # 易过拟合
        # self.sigmoid = nn.Softmax()  # 激活函数  # 对于多分类问题，需要使用softmax函数，各个类别互斥，总的类别概率为1。

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            # m.weight = m.weight.type(torch.FloatTensor).cuda()

    def forward(self, x, info):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        # x_out = x.cpu().detach().numpy()
        # print(x_out)
        # np.savetxt('x.txt', x_out, fmt='%0.5f')
        info = self.info_conv1(info)
        # sess = tf.Session()
        # print(info)
        x = torch.cat([x, info], dim=1)
        x = self.new_fc(x)
        # x = self.sigmoid(x)
        return x


def mobilenet_v2(pretrained=False, pretrainedModel=str, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        # pretrained_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],model_dir='/twx/pywork/pytorch/mobilenet/model_dir',progress=progress)
        pretrainedModel = torch.load(pretrainedModel)
        if isinstance(pretrainedModel, torch.nn.DataParallel):
            pretrainedModel = pretrainedModel.module
        # pretrained_dict = pretrainedModel.state_dict()  # pretrainedModel dict
        pretrained_dict = pretrainedModel  # pretrainedModel dict
        model_dict = model.state_dict()  # current model dict
        # print('====================================')
        # print(model_dict)
        # 筛除不加载的层结构；判断pretrained items是否在当前模型 current model dict里面
        dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print('*'*10, dict.keys())
        # # 删除不需要的权重
        # del_keys = ['info_conv1.weight', 'info_conv1.bias','new_fc.weight', 'new_fc.bias', 'features.18.0.weight',
        #             'features.18.1.weight','features.18.1.running_mean', 'features.18.1.running_var',
        #             'features.18.1.bias']
        # for k in del_keys:
        #     try:
        #         del dict[k]
        #     except:
        #         continue
        # 更新当前网络的结构字典
        model_dict.update(dict)
        model.load_state_dict(model_dict)
    return model


if __name__ == "__main__":

    # model = mobilenet_v2(pretrained=False, progress=True)
    # x = torch.rand(1,2,512,512)
    # # x = torch.rand(1,2,128,128)
    # info = torch.rand(1,7)
    # out = model(x, info)
    # print(out)
    model_dir = '/media/zhl/Model/0.86_test_fold4_mobilev2.pth'
    # print(pretrained_dict.keys())
    # print('===================================')
    # print(pretrained_dict.conv1)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = torch.load(model_dir)
        model = model.module
    print('===================================')
    print(model)
    # parm = {}  # 模型层的参数显示
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size())
    #     parm[name] = parameters.cpu().detach().numpy()
    #
    model_dict = model.state_dict()
    print('====================================')
    print(model_dict.keys)
    print('====================================')
    print(model_dict)

    # 筛除不加载的层结构
    dict = {k: v for k, v in model_dict.items() if k in model_dict}
    # 更新当前网络的结构字典
    model_dict.update(dict)
    model.load_state_dict(model_dict)
    print('====================================')
    print(model)

