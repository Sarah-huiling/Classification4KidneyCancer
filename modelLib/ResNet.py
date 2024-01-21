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
    def __init__(self, blocks, in_c=3, num_classes=2, expansion=4):
        super(ResNet, self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes=in_c, places=64)

        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)

        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)


        self.fc = nn.Linear(2048, num_classes)
        # self.fc = nn.Linear(4096, num_classes)
        # self.sigmoid = nn.Sigmoid()  # 激活函数
        self.sigmoid = nn.Softmax(dim=1)
        # 参数初始化
        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out')   # 何凯明Initialization
                nn.init.orthogonal_(m.weight)   # 正交初始化
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                # nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = x.mean([2, 3])
        x = x.view(x.size(0), -1)  # 因为卷积或者池化之后需要连接全连接层，所以需要把多维度的tensor展平成一维；
        # (batchsize，channels，x，y)变成(batchsize，channels*x*y)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


def ResNet50(in_c, num_classes):
    return ResNet([3, 4, 6, 3], in_c, num_classes)


def ResNet101(in_c, num_classes):
    return ResNet([3, 4, 23, 3], in_c, num_classes)


def ResNet152(in_c, num_classes):
    return ResNet([3, 8, 36, 3], in_c, num_classes)

# if __name__=='__main__':
#     #model = torchvision.models.resnet50()
#     model = ResNet50()
#     print(model)
#
#     input = torch.randn(1, 3, 224, 224)
#     out = model(input)
#     print(out.shape)
