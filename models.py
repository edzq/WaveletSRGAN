from torch import nn
from torch import cat
from torch import Tensor
from torch.nn import functional
from pywt import WaveletPacket2D



class WaveletTransform(nn.Module):
    def __init__(self, scale=1, dec=True, params_path='wavelet_weights_c2.pkl', transpose=True):
        super(WaveletTransform, self).__init__()

        self.scale = scale
        self.dec = dec
        self.transpose = transpose

        ks = int(2 ** scale)
        nc = 3 * ks * ks

        if dec:
            self.conv = nn.Conv2d(in_channels=3, out_channels=nc, kernel_size=ks, stride=ks, padding=0, groups=3,
                                  bias=False)
        else:
            self.conv = nn.ConvTranspose2d(in_channels=nc, out_channels=3, kernel_size=ks, stride=ks, padding=0,
                                           groups=3, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                f = open(params_path, 'rb')
                dct = pickle.load(f, encoding='bytes')
                f.close()
                key = 'rec{}'.format(ks).encode()
                m.weight.data = torch.from_numpy(dct[key])
                m.weight.requires_grad = False

    def forward(self, x):
        if self.dec:
            output = self.conv(x)
            if self.transpose:
                osz = output.size()
                # print(osz)
                output = output.view(osz[0], 3, -1, osz[2], osz[3]).transpose(1, 2).contiguous().view(osz)
        else:
            if self.transpose:
                xx = x
                xsz = xx.size()
                xx = xx.view(xsz[0], -1, 3, xsz[2], xsz[3]).transpose(1, 2).contiguous().view(xsz)
            output = self.conv(xx)
        return output



class Residual_Block(nn.Module):
    def __init__(self, c1, c2):
        super(Residual_Block, self).__init__()
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=1, padding=0, bias=True),
        )
        if c1 is not c2:
            self.identity_conv = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, padding=0,
                                           bias=True)
        else:
            self.identity_conv = None

    def forward(self, x):
        if self.identity_conv is not None:
            identity_x = self.identity_conv(x)
        else:
            identity_x = x
        return identity_x + self.residual_block(x)


class Wavelet_Prediction_Block(nn.Module):
    def __init__(self, c1, c2, c3):
        super(Wavelet_Prediction_Block, self).__init__()
        self.block = nn.Sequential(
            Residual_Block(c1=c1, c2=c2),
            Residual_Block(c1=c2, c2=c2),
            Residual_Block(c1=c2, c2=c3),
            Residual_Block(c1=c3, c2=c3)
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, n):
        super(Generator, self).__init__()
        self.n = n
        self.scale = int(2 ** n)
        self.Nw = int(4 ** n)

        self.embedding_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True),
            Residual_Block(c1=128,c2=256),
            Residual_Block(c1=256, c2=256),
            Residual_Block(c1=256, c2=512),
            Residual_Block(c1=512, c2=512),
            Residual_Block(c1=512, c2=1024),
            Residual_Block(c1=1024, c2=1024)
        )

        self.wave = nn.ModuleList(
            [Wavelet_Prediction_Block(c1=1024, c2=32, c3=64) for _ in range(2 ** (self.n + 1) - 1)]
        )

    def forward(self, x):
        out = self.embedding_net(x)
        out = self.wave[0](out)
        for i in range(1, len(self.wave)):
            out = cat((out, self.wave[i](x)), 1)
        out = self.wave[0](x)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )
    def forward(self, x):
        return functional.sigmoid(self.net(x).view(x.size[0]))
