""" Convolutional Autoencoder for video anomaly detection. """
import torch
import torch.nn as nn
import torch.nn.functional as F


class STAE(nn.Module):
    """ Implemenets spatial-tempoal autoencoder without the prediction
    branch.
    """
    def __init__(self, in_channels, negative_slope=1e-2):
        super(STAE, self).__init__()
        self.conv1 = self._conv_3d_block(in_channels, 32, negative_slope)
        self.conv2 = self._conv_3d_block(32, 48, negative_slope)
        self.conv3 = self._conv_3d_block(48, 64, negative_slope)
        self.conv4 = self._conv_3d_block(64, 64, negative_slope, False)
        self.deconv1_1 = self._deconv_3d_block(64, 48, negative_slope)
        self.deconv2_1 = self._deconv_3d_block(48, 32, negative_slope)
        self.deconv3_1 = self._deconv_3d_block(32, 32, negative_slope)
        self.conv5_1 = nn.Conv3d(32, in_channels, kernel_size=3, padding=1)
        self.deconv1_2 = self._deconv_3d_block(64, 48, negative_slope)
        self.deconv2_2 = self._deconv_3d_block(48, 32, negative_slope)
        self.deconv3_2 = self._deconv_3d_block(32, 32, negative_slope)
        self.conv5_2 = nn.Conv3d(32, in_channels, kernel_size=3, padding=1)

    def encode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x

    def decode_reconstruction(self, x):
        """ Decoder for the reconstruction branch that takes encoder output
        as input. """
        x = self.deconv1_1(x)
        x = self.deconv2_1(x)
        x = self.deconv3_1(x)
        x = F.sigmoid(self.conv5_1(x))

        return x

    def decode_prediction(self, x):
        """ Decoder for the prediction branch that takes encoder output
        as input. """
        x = self.deconv1_2(x)
        x = self.deconv2_2(x)
        x = self.deconv3_2(x)
        x = F.sigmoid(self.conv5_2(x))

        return x

    def forward(self, x):
        """ Forward and output reconstruction and prediction results.
        Input:
            x: (batch_size, channels, depth, height, width).
        Output:
            recon: reconstructions with the same shape as input.
            pred: predictions with the same shape as input.
        """
        x = self.encode(x)
        recon = self.decode_reconstruction(x)
        pred = self.decode_prediction(x)

        return recon, pred

    def _conv_3d_block(self, in_channels, out_channels, negative_slope=1e-2,
                       pool=True):
        """ 3D convolution for encoder part.
        In pytorch the input to nn.Conv3d has the following arrangement:
        [batch_size, channels, depth, height, width]. """
        if pool:
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.MaxPool3d(kernel_size=2, stride=2)
            )
        else:
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(negative_slope=negative_slope)
            )

    def _deconv_3d_block(self, in_channels, out_channels, negative_slope=1e-2):
        """ 3D deconvolution for decoder part.
        Two implementations are possible:

        nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4,
                           padding=1, stride=2),
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3,
                           padding=1, stride=2, output_padding=1).

        Here we use the first one.
        """
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4,
                               padding=1, stride=2),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope)
        )


class PredictionLoss(nn.Module):
    """ Implements weight-decreasing prediction loss. """
    def __init__(self, size_average=True):
        super(PredictionLoss, self).__init__()
        self.size_average = size_average

    def forward(self, predicted_frames, current_frames):
        """ Calculates the prediction loss between the current frames and
        the predicted frames.
        The input volumes are organized as
            [batch_size, channels, volume_size, height, width],
        where volume_size is the number of frames in each video cube.
        """
        T = predicted_frames.size(2)
        loss = (predicted_frames - current_frames).pow(2)
        for idx in range(T):
            loss[:, :, idx, :, :] *= (T-idx-1) / (T**2)
        loss = loss.sum()
        if self.size_average:
            loss = loss / predicted_frames.size(0)

        return loss
