import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init

import math

try:
    from models.blocks import FRN, ActFirstResBlk
except:
    from blocks import FRN, ActFirstResBlk


class Discriminator(nn.Module):
    """Discriminator: (image x, domain y) -> (logit out)."""
    def __init__(self, image_size=256, num_domains=2, max_conv_dim=1024):
        super(Discriminator, self).__init__()
        dim_in = 64 if image_size < 256 else 32
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(image_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ActFirstResBlk(dim_in, dim_in, downsample=False)]
            blocks += [ActFirstResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)] #1x1 conv
        self.main = nn.Sequential(*blocks)

        self.apply(weights_init('kaiming'))

    def forward(self, x, y):
        """
        Inputs:
            - x: images of shape (batch, 3, image_size, image_size).
            - y: domain indices of shape (batch).
        Output:
            - out: logits of shape (batch).
        """
        out = self.main(x)
        feat = out
        out = out.view(out.size(0), -1)                          # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]                                         # (batch)
        return out, feat

    def _initialize_weights(self, mode='fan_in'):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

class Discriminator_PatchGAN(nn.Module):
    def __init__(self, in_channels=3, num_domains=2):
        super(Discriminator_PatchGAN, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            #*discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256, stride=1),
            #*discriminator_block(256, 512),
            #nn.ZeroPad2d((1, 0, 1, 0)),
            #nn.Conv2d(512, 1, 4, padding=1, bias=False)
            nn.Conv2d(256, num_domains, 4, padding=1, bias=False) # output = N * num_domains * 18 * 18
        )
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, img_A, y):
        # Concatenate image and condition image by channels to produce input
        out = self.model(img_A) # (batch, num_domains, 18, 18)
        feat = out.view(out.size(0), out.size(1), -1) # (batch, num_domains, 18*18)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        logit = feat[idx, y, :] # (batch, 18*18)
        #logit = self.softmax(feat[idx]) ## '''modified!!'''

        return logit, out



def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun


# if __name__ == '__main__':
#     D = Discriminator(64, 10)
#     x_in = torch.randn(4, 3, 64, 64)
#     y_in = torch.randint(0, 10, size=(4, ))
#     print(y_in)
#     out, feat = D(x_in, y_in)
#     print(out)
#     print(out.shape, feat.shape)

# if __name__ == '__main__':
#     D = Discriminator(3, 10)
#     x_in = torch.randn(4, 3, 80, 80)
#     y_in = torch.randint(0, 10, size=(4, ))
#     print(y_in)
#     logit, out = D(x_in, y_in)
#     print(logit.shape, out.shape)