import torch
from torch import nn
import torch.nn.functional as nnf
import torchvision.models as models


class UpConv(nn.Module):
    """
    Up-sample feature 1 and concat it with feature 2, and conv twice.
    """
    def __init__(self, f1_ch, f2_ch, out_ch):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(f1_ch, f1_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(f1_ch + f2_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, f1, f2):
        _out = self.up(f1)
        _out = self.conv(torch.cat([f2, _out], dim=1))
        return _out


"""
Taken from https://github.com/shashankvkt/video_object_segmentation
"""


class Initializer(nn.Module):
    def __init__(self, in_ch=3):
        super(Initializer, self).__init__()
        self.new_layer = nn.Conv2d(in_channels=in_ch * 2, out_channels=64, kernel_size=3, padding=1)
        self.pretrained_model = models.vgg16(pretrained=True)
        self.model = nn.Sequential(*list(self.pretrained_model.features.children())[2:31])

        self.c0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.h0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.new_layer(inputs))
        x = self.model(x)
        c0 = self.relu(self.c0(x))
        h0 = self.relu(self.h0(x))
        return c0, h0


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pretrained_model = models.vgg16(pretrained=True)
        self.model = nn.Sequential(*list(self.pretrained_model.features.children())[0:31])
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.model(inputs)
        x = self.relu(x)
        return x


class ConvLSTMCell(nn.Module):
    """
    Code taken and modified from https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, 3, padding=3 // 2)

    def forward(self, input_, hidden_state, cell_state):
        # generate empty prev_state, if None is provided
        prev_hidden = hidden_state
        prev_cell = cell_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non-linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non-linearity
        cell_gate = nnf.relu(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * nnf.relu(cell)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, out_ch=3):
        super(Decoder, self).__init__()

        self.interp = nnf.interpolate
        self.relu = nn.ReLU()
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1)

        self.conv = nn.Conv2d(64, out_ch, kernel_size=5, padding=2)

        self.softmax = nn.Softmax()

    def forward(self, inputs):
        x = self.deconv1(inputs)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.relu(x)
        x = self.deconv4(x)
        x = self.relu(x)
        x = self.deconv5(x)
        x = self.relu(x)

        x = self.conv(x)
        x = self.softmax(x)

        return x


class LSTMVOS(nn.Module):
    def __init__(self):
        super(LSTMVOS, self).__init__()
        self.initializer = Initializer()
        self.encoder = Encoder()
        self.convlstm = ConvLSTMCell(input_size=512, hidden_size=512)
        self.decoder = Decoder()

    def forward(self, init_rgb, init_mask, rgb_data):
        predicted_mask = []
        c0, h0 = self.initializer(torch.cat((init_rgb, init_mask), 1))
        for i in range(5):
            rgb_frame = rgb_data[:, i, :, :, :]
            x_tilda = self.encoder(rgb_frame)
            c_next, h_next = self.convlstm(x_tilda, h0, c0)
            output = self.decoder(h_next)
            c0 = c_next
            h0 = h_next
            predicted_mask.append(output)
        predicted_mask = torch.stack(predicted_mask, dim=1)
        return predicted_mask


class EncoderWithSC(nn.Module):
    def __init__(self, pretrained=True):
        super(EncoderWithSC, self).__init__()
        self.encoder = models.resnet18(pretrained=pretrained)

    def forward(self, inputs):
        f0 = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(inputs)))
        f1 = self.encoder.layer1(self.encoder.maxpool(f0))
        f2 = self.encoder.layer2(f1)
        f3 = self.encoder.layer3(f2)
        f4 = self.encoder.layer4(f3)
        return f0, f1, f2, f3, f4


class DecoderWithSC(nn.Module):
    def __init__(self, out_ch=3):
        super(DecoderWithSC, self).__init__()
        self.relu = nn.ReLU()
        self.decoder0 = UpConv(512, 256, 256)
        self.decoder1 = UpConv(256, 128, 128)
        self.decoder2 = UpConv(128, 64, 64)
        self.decoder3 = UpConv(64, 64, 64)
        self.last = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, out_ch, kernel_size=5, padding=2),
            nn.Softmax(dim=1)
        )

    def forward(self, x, f0, f1, f2, f3):
        """
        replace the last feature with x
        """
        _out = self.decoder0(x, f3)
        _out = self.decoder1(_out, f2)
        _out = self.decoder2(_out, f1)
        _out = self.decoder3(_out, f0)
        _out = self.last(_out)
        return _out


class LSTMVOSWithSC(nn.Module):
    def __init__(self):
        super(LSTMVOSWithSC, self).__init__()
        self.initializer = Initializer()
        self.encoder = EncoderWithSC()
        self.convlstm = ConvLSTMCell(input_size=512, hidden_size=512)
        self.decoder = DecoderWithSC()

    def forward(self, init_rgb, init_mask, rgb_data):
        """
        Args:
            init_rgb (torch.Tensor): size of (B, C, H, W)
            init_mask (torch.Tensor): size of (B, C, H, W)
            rgb_data (torch.Tensor): size of (B, N=5, C, H, W)
        """
        predicted_mask = []
        c0, h0 = self.initializer(torch.cat((init_rgb, init_mask), 1))
        for i in range(5):
            rgb_frame = rgb_data[:, i, :, :, :]
            f0, f1, f2, f3, f4 = self.encoder(rgb_frame)
            x_tilda = f4
            c_next, h_next = self.convlstm(x_tilda, h0, c0)
            output = self.decoder(h_next, f0, f1, f2, f3)
            c0 = c_next
            h0 = h_next
            predicted_mask.append(output)
        predicted_mask = torch.stack(predicted_mask, dim=1)
        return predicted_mask


class LSTMVOSWithSCTestTime(LSTMVOSWithSC):
    def __init__(self):
        super(LSTMVOSWithSCTestTime, self).__init__()
        self.last_c = None
        self.last_h = None

    def test_init(self, init_rgb, init_mask):
        """
        Args:
            init_rgb (torch.Tensor): size of (B, C, H, W)
            init_mask (torch.Tensor): size of (B, C, H, W)
        """
        self.last_c, self.last_h = self.initializer(torch.cat((init_rgb, init_mask), 1))

    def test(self, single_rgb):
        """
        Args:
            single_rgb (torch.Tensor): size of (B, C, H, W)
        Returns:
            torch.Tensor: size of (B, C, H, W)
        """
        f0, f1, f2, f3, f4 = self.encoder(single_rgb)
        x_tilda = f4
        next_c, next_h = self.convlstm(x_tilda, self.last_h, self.last_c)
        output = self.decoder(next_h, f0, f1, f2, f3)
        self.last_c, self.last_h = next_c, next_h
        return output

