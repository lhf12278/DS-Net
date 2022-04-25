import torch.nn as nn
import torch
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


def nn_conv2d(im):
    conv_op = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    sobel_kernel = np.array([[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 3, 3, 3))
    conv_op.weight.data = torch.from_numpy(sobel_kernel).cuda()
    edge_detect = conv_op(im)
    edge_detect = edge_detect.detach()
    edge_detect = torch.cat((edge_detect, edge_detect, edge_detect), 1)
    return edge_detect


class PALayer(nn.Module):
    def __init__(self, channel=64):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel=64):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class ExtractorSobel(nn.Module):
    def __init__(self):
        super(ExtractorSobel, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(64, 64 // 8, 3, padding=(3 // 2), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64 // 8, 64, 3, padding=(3 // 2), bias=True)
        )
        self.pa = PALayer()
        self.ca = CALayer()
        self.conv11 = nn.Sequential(
            nn.Conv2d(2 * 64, 64, 1, padding=(1 // 2), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=(3 // 2), bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = self.extractor(x) + x
        out_ca = self.ca(res)
        out_pa = self.pa(out_ca)
        out_hunhe = torch.cat((out_pa, out_ca), 1)
        Ex_out = self.conv11(out_hunhe) * out_pa + out_pa
        return Ex_out


class ExtractorDeHaze(nn.Module):
    def __init__(self):
        super(ExtractorDeHaze, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(64, 64 // 8, 3, padding=(3 // 2), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64 // 8, 64, 3, padding=(3 // 2), bias=True)
        )
        self.pa = PALayer()
        self.ca = CALayer()
        self.conv11 = nn.Sequential(
            nn.Conv2d(2 * 64, 64, 1, padding=(1 // 2), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=(3 // 2), bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = self.extractor(x) + x
        out_ca = self.ca(res)
        out_pa = self.pa(out_ca)
        out_hunhe = torch.cat((out_pa, out_ca), 1)
        Ex_out = self.conv11(out_hunhe) * out_pa + out_pa

        return Ex_out


class ReconsModel(nn.Module):
    def __init__(self):
        super(ReconsModel, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=(3 // 2), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=(3 // 2), bias=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(64, 32, 1, padding=(1 // 2), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1, padding=(1 // 2), bias=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(64, 32, 5, padding=(5 // 2), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 5, padding=(5 // 2), bias=True)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(64, 3, 1, padding=(1 // 2), bias=True)
        )

        self.cca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(9, 64 // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64 // 16, 9, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])
        self.palayer = nn.Sequential(
            nn.Conv2d(3, 9, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(9, 3, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        post_precess = [
            default_conv(3, 64, 3),
            default_conv(64, 3, 3)]
        self.post = nn.Sequential(*post_precess)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        w = self.cca(torch.cat([b1, b2, b3], dim=1))
        w = w.view(-1, 3, 3)[:, :, :, None, None]
        out = w[:, 0, :, :] * b1 + w[:, 1, :, :] * b2 + w[:, 2, :, :] * b3
        out = self.palayer(out) * out
        result = self.post(out)

        return result


class Reconssobel(nn.Module):
    def __init__(self):
        super(Reconssobel, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(64, 3, 1, padding=(1 // 2), bias=True)
        )

    def forward(self, x):
        result = self.fusion(x)
        return result


class InterActed(nn.Module):
    def __init__(self):
        super(InterActed, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.trans = nn.Sequential(
            nn.Linear(256, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 64, bias=True),
            nn.Softmax()
        )

        self.enhance_layer = nn.Sequential(
            nn.Conv2d(64, 64//16, 3, padding=(3 // 2), bias=True),
            nn.Conv2d(64//16, 64, 3, padding=(3 // 2), bias=True)
        )

        self.fu = nn.Conv2d(128, 64, 1, padding=(1 // 2), bias=True)

        self.ppa = PALayer()

    def forward(self, x_haze, x_sobel):
        x_hunhe = torch.cat((x_haze, x_sobel), 1)
        F = torch.cat((self.avg_pool(x_hunhe), self.max_pool(x_hunhe)), 1)
        F = F.view(F.size()[0], -1)

        P = self.trans(F)
        P = torch.unsqueeze(P, 2)
        P = torch.unsqueeze(P, 3)

        P_ = torch.full_like(P, 1) - P
        x1 = x_haze * P + self.enhance_layer(x_haze * P_) + x_haze
        x1 = self.ppa(x1) + x1 + x_haze
        x2 = self.enhance_layer(x_sobel * P_) + x_sobel + x_sobel * P
        x2 = self.ppa(x2) + x2 + x_sobel
        x_up = self.fu(torch.cat((x_haze, x1), 1))
        x_down = self.fu(torch.cat((x_sobel, x2), 1))

        return x_up, x_down


class DeHaze(nn.Module):
    def __init__(self):
        super(DeHaze, self).__init__()
        self.preSobel = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=(3 // 2), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=(3 // 2), bias=True)
        )
        self.prehaze = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=(3 // 2), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=(3 // 2), bias=True)
        )

        self.EsDehaze = nn.ModuleList()
        for d in range(9):  # 9
            self.EsDehaze.append(ExtractorDeHaze())

        self.EsSobelhaze = nn.ModuleList()
        for d in range(9):  # 9
            self.EsSobelhaze.append(ExtractorSobel())

        self.r = nn.Sigmoid()

        self.IA = InterActed()

        self.c = nn.Conv2d(9*64, 64, 1, padding=0, bias=True)
        self.redehaze = ReconsModel()
        self.resobel = Reconssobel()

        self.conv11 = nn.Sequential(
            nn.Conv2d(2*64, 64, 1, padding=(1 // 2), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=(3 // 2), bias=True),
            nn.Sigmoid()
        )

    def forward(self, haze, gt):
        SobelHaze = nn_conv2d(haze)
        SobelClear = nn_conv2d(gt)
        out_sobel = self.preSobel(SobelHaze)
        out_dehaze = self.prehaze(haze)

        i = 0
        Sobelhaze_outs = []
        Dehaze_outs = []
        while i < 9:
            out_sobel = self.EsSobelhaze[i](out_sobel)
            out_dehaze = self.EsDehaze[i](out_dehaze)
            out_dehaze, out_sobel = self.IA(out_dehaze, out_sobel)

            Sobelhaze_outs.append(out_sobel)
            Dehaze_outs.append(out_dehaze)
            i += 1

        out_sobel = self.c(torch.cat(Sobelhaze_outs, 1))
        out_dehaze = self.c(torch.cat(Dehaze_outs, 1))

        res_dehaze = self.redehaze(out_dehaze)
        res_sobel = self.resobel(out_sobel)

        return SobelClear, res_dehaze, res_sobel

