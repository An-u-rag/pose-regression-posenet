import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
torch.set_printoptions(threshold=10_000)


def init(key, module, weights=None):
    if weights == None:
        return module

    # initialize bias.data: layer_name_1 in weights
    # initialize  weight.data: layer_name_0 in weights
    module.bias.data = torch.from_numpy(weights[(key + "_1").encode()])
    module.weight.data = torch.from_numpy(weights[(key + "_0").encode()])

    return module


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, key=None, weights=None):
        super(InceptionBlock, self).__init__()

        # Use 'key' to load the pretrained weights for the correct InceptionBlock

        # 1x1 conv branch
        self.b1 = nn.Sequential(
            init(f'inception_{key}/1x1', nn.Conv2d(in_channels,
                 n1x1, kernel_size=1, stride=1, padding=0), weights),
            nn.ReLU()
        )

        # 1x1 -> 3x3 conv branch
        self.b2 = nn.Sequential(
            init(f'inception_{key}/3x3_reduce', nn.Conv2d(in_channels, n3x3red,
                                                          kernel_size=1, stride=1, padding=0), weights),
            nn.ReLU(),
            init(f'inception_{key}/3x3', nn.Conv2d(n3x3red,
                 n3x3, kernel_size=3, stride=1, padding=1), weights),
            nn.ReLU()
        )

        # 1x1 -> 5x5 conv branch
        self.b3 = nn.Sequential(
            init(f'inception_{key}/5x5_reduce', nn.Conv2d(in_channels, n5x5red,
                                                          kernel_size=1, stride=1, padding=0), weights),
            nn.ReLU(),
            init(f'inception_{key}/5x5', nn.Conv2d(n5x5red,
                 n5x5, kernel_size=5, stride=1, padding=2), weights),
            nn.ReLU()
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            init(f'inception_{key}/pool_proj', nn.Conv2d(in_channels, pool_planes,
                                                         kernel_size=1, stride=1, padding=0), weights),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.b1(x)
        x2 = self.b2(x)
        x3 = self.b3(x)
        x4 = self.b4(x)
        x5 = torch.concat((x1, x2, x3, x4), dim=1)
        return x5


class LossHeader(nn.Module):
    def __init__(self, in_channels, key, weights=None):
        super(LossHeader, self).__init__()

        self.auxlayers = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3, padding=0),
            nn.ReLU(),
            init(f'{key}/conv', nn.Conv2d(in_channels, 128,
                 kernel_size=1, stride=1, padding=0), weights),
            nn.ReLU(),
            nn.Flatten(),
            init(f'{key}/fc', nn.Linear(2048, 1024), weights),
            nn.Dropout(p=0.7)
        )

        self.auxfc1 = nn.Linear(1024, 3)
        self.auxfc2 = nn.Linear(1024, 4)

    def forward(self, x):
        out = self.auxlayers(x)
        xyz = self.auxfc1(out)
        wpqr = self.auxfc2(out)
        return xyz, wpqr


class PoseNet(nn.Module):
    def __init__(self, load_weights=True):
        super(PoseNet, self).__init__()

        # Load pretrained weights file
        if load_weights:
            print("Loading pretrained InceptionV1 weights...")
            file = open('pretrained_models/places-googlenet.pickle', "rb")
            weights = pickle.load(file, encoding="bytes")
            file.close()
        # Ignore pretrained weights
        else:
            weights = None

        self.pre_layers = nn.Sequential(
            # Example for defining a layer and initializing it with pretrained weights
            init('conv1/7x7_s2', nn.Conv2d(3, 64,
                 kernel_size=7, stride=2, padding=3), weights),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # LRN
            nn.LocalResponseNorm(size=5),
            init('conv2/3x3_reduce', nn.Conv2d(64, 64,
                 kernel_size=1, stride=1, padding=0), weights),
            nn.ReLU(),
            init('conv2/3x3', nn.Conv2d(64, 192,
                 kernel_size=3, stride=1, padding=1), weights),
            nn.ReLU(),
            # LRN
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Example for InceptionBlock initialization
        self._3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32, "3a", weights)
        self._3b = InceptionBlock(
            256, 128, 128, 192, 32, 96, 64, "3b", weights)
        self._3maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64, "4a", weights)
        self._4b = InceptionBlock(
            512, 160, 112, 224, 24, 64, 64, "4b", weights)
        self._4c = InceptionBlock(
            512, 128, 128, 256, 24, 64, 64, "4c", weights)
        self._4d = InceptionBlock(
            512, 112, 144, 288, 32, 64, 64, "4d", weights)
        self._4e = InceptionBlock(
            528, 256, 160, 320, 32, 128, 128, "4e", weights)
        self._4maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._5a = InceptionBlock(
            832, 256, 160, 320, 32, 128, 128, "5a", weights)
        self._5b = InceptionBlock(
            832, 384, 192, 384, 48, 128, 128, "5b", weights)

        self.postlayers = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 2048),
            nn.Dropout(p=0.4)
        )
        self.fc1 = nn.Linear(2048, 3)
        self.fc2 = nn.Linear(2048, 4)

        self.aux1 = LossHeader(512, "loss1", weights)
        self.aux2 = LossHeader(528, "loss2", weights)

        self.relu = nn.ReLU()
        print("PoseNet model created!")

    def forward(self, x):
        x1 = self.pre_layers(x)
        x_3a = self._3a(x1)
        x_3b = self._3b(x_3a)
        x_3b = self._3maxpool(x_3b)
        x_3b = self.relu(x_3b)
        x_4a = self._4a(x_3b)
        x_4b = self._4b(x_4a)
        loss1_xyz, loss1_wpqr = self.aux1(x_4a)
        x_4c = self._4c(x_4b)
        x_4d = self._4d(x_4c)
        x_4e = self._4e(x_4d)
        loss2_xyz, loss2_wpqr = self.aux2(x_4d)
        x_4e = self._4maxpool(x_4e)
        x_4e = self.relu(x_4e)
        x_5a = self._5a(x_4e)
        x_5b = self._5b(x_5a)
        out = self.postlayers(x_5b)
        loss3_xyz = self.fc1(out)
        loss3_wpqr = self.fc2(out)

        if self.training:
            return loss1_xyz, \
                loss1_wpqr, \
                loss2_xyz, \
                loss2_wpqr, \
                loss3_xyz, \
                loss3_wpqr
        else:
            return loss3_xyz, \
                loss3_wpqr


class PoseLoss(nn.Module):

    def __init__(self, w1_xyz, w2_xyz, w3_xyz, w1_wpqr, w2_wpqr, w3_wpqr):
        super(PoseLoss, self).__init__()

        # Influence
        self.w1_xyz = w1_xyz
        self.w2_xyz = w2_xyz
        self.w3_xyz = w3_xyz

        # Quaternion Betas
        self.w1_wpqr = w1_wpqr
        self.w2_wpqr = w2_wpqr
        self.w3_wpqr = w3_wpqr

    def forward(self, p1_xyz, p1_wpqr, p2_xyz, p2_wpqr, p3_xyz, p3_wpqr, poseGT):
        # First 3 entries of poseGT are ground truth xyz, last 4 values are ground truth wpqr
        batch_size = poseGT.size()[0]

        loss = 0
        for i in range(p1_xyz.size()[0]):
            p_gt = poseGT[i, :3]
            q_gt = poseGT[i, 3:]
            q_gtn = F.normalize(q_gt, p=2.0, dim=0)

            # Loss 1
            loss1_xyz = torch.norm((p1_xyz[i] - p_gt))
            loss1_wpqr = self.w1_wpqr * (torch.norm(p1_wpqr[i] - q_gtn))
            loss1 = loss1_xyz + loss1_wpqr

            # Loss 2
            loss2_xyz = torch.norm((p2_xyz[i] - p_gt))
            loss2_wpqr = (self.w2_wpqr * (torch.norm(p2_wpqr[i] - q_gtn)))
            loss2 = loss2_xyz + loss2_wpqr

            # Loss 3
            loss3_xyz = torch.norm((p3_xyz[i] - p_gt))
            loss3_wpqr = (self.w3_wpqr * (torch.norm(p3_wpqr[i] - q_gtn)))
            loss3 = loss3_xyz + loss3_wpqr

            # Total Loss
            loss += (self.w1_xyz * loss1) + \
                (self.w2_xyz * loss2) + (self.w3_xyz * loss3)

        loss = loss/batch_size

        return loss
