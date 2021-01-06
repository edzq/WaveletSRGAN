from torch import nn


class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()
        self.wavelet

    def forward(self, out_labels, out_coefficients, lr_images, target_images):
        target_coefficients = self.dwt(target_images)
        out_images = self.idwt((lr_images, out_coefficients))
        l_content = self.mse(out_images, target_images)
        l_wavelet = self.mse(out_coefficients, target_coefficients)
        l_adversarial =

