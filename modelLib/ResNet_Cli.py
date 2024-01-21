import torch
import torch.nn as nn
import torchvision

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

__all__ = ['ResNet50', 'ResNet101', 'ResNet152']


def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    # num_classes要分类的类别
    def __init__(self, blocks, in_c, num_classes=2, info_num=8, expansion=4):
        super(ResNet, self).__init__()
        self.expansion = expansion
        # self.info_num = info_num

        # self.conv1 = Conv1(in_planes=88, places=64)  # 输入通道
        self.conv1 = Conv1(in_planes=in_c, places=64)  # 输入通道

        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)

        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)  #

        self.info_conv1 = nn.Linear(info_num, 256)  # 10个临床特征，上采样为256

        self.fc = nn.Linear(2048 + 256, num_classes)  # 全连接层
        # self.sigmoid = nn.Sigmoid()  # 激活函数
        # self.sigmoid = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x, info):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = x.mean([2, 3])
        info = self.info_conv1(info)
        x = torch.cat([x, info], dim=1)
        x = self.fc(x)
        return x


def ResNet50(in_c, num_classes, info_num):
    return ResNet([3, 4, 6, 3], in_c, num_classes,info_num)


def ResNet101(in_c, num_classes,info_num):
    return ResNet([3, 4, 23, 3], in_c, num_classes,info_num)


def ResNet152(in_c, num_classes,info_num):
    return ResNet([3, 8, 36, 3], in_c, num_classes,info_num)


def ResNet50_pre(pretrained=False, pretrainedModel=str, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet50(**kwargs)
    if pretrained:
        pretrainedModel = torch.load(pretrainedModel)
        if isinstance(pretrainedModel, torch.nn.DataParallel):
            pretrainedModel = pretrainedModel.module
        pretrained_dict = pretrainedModel.state_dict()  # pretrainedModel dict
        # pretrained_dict = pretrainedModel  # pretrainedModel dict
        model_dict = model.state_dict()  # current model dict
        # print('====================================')
        # print(model_dict)
        # 筛除不加载的层结构；判断pretrained items是否在当前模型 current model dict里面
        dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # print('*'*10, dict.keys())
        # # 删除不需要的权重
        # del_keys = ['info_conv1.weight', 'info_conv1.bias','new_fc.weight', 'new_fc.bias', 'features.18.0.weight',
        #             'features.18.1.weight','features.18.1.running_mean', 'features.18.1.running_var',
        #             'features.18.1.bias']
        for k in ['fc.weight']:
            try:
                del dict[k]
            except:
                continue
        # 更新当前网络的结构字典
        model_dict.update(dict)
        model.load_state_dict(model_dict)
    return model

if __name__=='__main__':
    # model = torchvision.models.resnet50()
    # model = ResNet50()
    model = torch.load('/media/zhl/ResearchData/20221028华西金玉梅直肠癌疗效评估/Result/DL/peritumor_resNet1/model/0.99_test_fold4_resNet50.pth')
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)
