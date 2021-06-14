from math import exp

import torch
from torch import nn
from torchvision.models.vgg import vgg16
import torch.nn.functional as F


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        # loss_network = nn.Sequential(
        #     nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     *list(vgg.features)[:31]
        # ).eval()

        vgg = vgg16(pretrained=True)
        vgg = VGG(vgg.features[:23]).eval()
        
        self.vgg16 = vgg
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)

        out_features = self.vgg16(out_images)
        target_features = self.vgg16(target_images)
        # style loss
        style_grams = [gram_matrix(x) for x in target_features]
        out_grams = [gram_matrix(x) for x in out_features]
        style_loss = 0
        for a, b in zip(out_grams, style_grams):
            style_loss += self.mse_loss(a, b)
        # Perception Loss
        perception_loss = 0
        for a, b in zip(out_features, target_features):
            perception_loss += self.mse_loss(a, b)
        # perception_loss = self.mse_loss(out_features[2], target_features[2])
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)

        # TV Loss
        tv_loss = self.tv_loss(out_images)
        # return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss + 0.01 * sequence_loss
        # return image_loss + 0.001 * adversarial_loss + 0.06 * perception_loss + 2e-8 * tv_loss # + 0.01 * sequence_loss
        # return image_loss
        # return image_loss + 0.006 * perception_loss
        # return image_loss + 0.1 * perception_loss + 0.001 * adversarial_loss
        return 1 * image_loss + 1 * perception_loss + 1e6 * style_loss + 2e-8 * tv_loss


class CNNLoss(nn.Module):
    def __init__(self):
        super(CNNLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        vgg = VGG(vgg.features[:23]).eval()
        
        self.vgg16 = vgg
        self.mse_loss = nn.MSELoss()
        self.l1_loss = CharbonnierLoss()
        self.tv_loss = TVLoss()
        self.ssim_loss = SsimLoss()

    def forward(self, out_images, target_images):
        out_features = self.vgg16(out_images)
        target_features = self.vgg16(target_images)
        # style loss
        style_grams = [gram_matrix(x) for x in target_features]
        out_grams = [gram_matrix(x) for x in out_features]
        style_loss = 0
        for a, b in zip(out_grams, style_grams):
            style_loss += self.mse_loss(a, b)
        # Perception Loss
        perception_loss = 0
        for a, b in zip(out_features, target_features):
            perception_loss += self.mse_loss(a, b)
        # perception_loss = self.mse_loss(out_features[2], target_features[2])
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        #image_loss = self.l1_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        # SSIM Loss
        ssim_loss = self.ssim_loss(out_images, target_images)
        # return image_loss
        #return 1 * image_loss
        return 1 * image_loss + 1 * perception_loss + 1e6 * style_loss
        # return 1 * image_loss + 1e-2 * perception_loss + 1e4 * style_loss + 2e-6 * tv_loss
        # return 1 * image_loss + 1e-2 * perception_loss + 1e4 * style_loss + 2e-6 * tv_loss + 1 * ssim_loss
        # return 1 * image_loss + 1 * ssim_loss


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class VGG(nn.Module):

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
        for p in self.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        outs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                outs.append(x)
        return outs


def gram_matrix(y):
    b, ch , h, w = y.shape
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram



class SsimLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SsimLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1 - _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):

    if len(img1.size()) == 3:
        img1 = torch.stack([img1], dim=0)
        img2 = torch.stack([img2], dim=0)

    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
