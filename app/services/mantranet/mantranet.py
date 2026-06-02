"""
ManTraNet - Manipulation Tracing Network (PyTorch)
Faithful port from: https://github.com/RonyAbecidan/ManTraNet-pytorch

Detects image forgery at pixel level:
  - Splicing, copy-move, removal, enhancement
  - Outputs a forgery likelihood heatmap (H×W, values 0–1)

Architecture:
  1. IMTFE  — Image Manipulation Trace Feature Extractor
     (Bayar filters + SRM filters + learned conv → deep feature stack)
  2. AnomalyDetector — Local anomaly detection via multi-scale Z-pooling + ConvLSTM
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# ── Utility Functions ──────────────────────────────────────────────────────────

def hardsigmoid(T):
    """Keras-compatible hard sigmoid."""
    T_0 = T
    T = 0.2 * T_0 + 0.5
    T[T_0 < -2.5] = 0
    T[T_0 > 2.5] = 1
    return T


def reflect(x, minx, maxx):
    """Reflects an array around two points for symmetric padding."""
    rng = maxx - minx
    double_rng = 2 * rng
    mod = np.fmod(x - minx, double_rng)
    normed_mod = np.where(mod < 0, mod + double_rng, mod)
    out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)


def symm_pad(im, padding):
    """Symmetric (reflect) padding matching TensorFlow behavior."""
    h, w = im.shape[-2:]
    left, right, top, bottom = padding
    x_idx = np.arange(-left, w + right)
    y_idx = np.arange(-top, h + bottom)
    x_pad = reflect(x_idx, -0.5, w - 0.5)
    y_pad = reflect(y_idx, -0.5, h - 0.5)
    xx, yy = np.meshgrid(x_pad, y_pad)
    return im[..., yy, xx]


def batch_norm(X, eps=0.001):
    """Custom batch norm matching TF/Keras behavior (instance-level stats)."""
    N, C, H, W = X.shape
    device = X.device
    mean = X.mean(axis=(0, 2, 3)).to(device)
    variance = ((X - mean.view((1, C, 1, 1))) ** 2).mean(axis=(0, 2, 3)).to(device)
    X = (X - mean.reshape((1, C, 1, 1))) * 1.0 / torch.pow(
        (variance.view((1, C, 1, 1)) + eps), 0.5
    )
    return X.to(device)


# ── ConvLSTM ───────────────────────────────────────────────────────────────────

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )
        self.sigmoid = hardsigmoid

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_c, cc_o = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = self.sigmoid(cc_i)
        f = self.sigmoid(cc_f)
        c_next = f * c_cur + i * torch.tanh(cc_c)
        o = self.sigmoid(cc_o)
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
        )


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super().__init__()
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all(isinstance(elem, tuple) for elem in kernel_size))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

        kernel_size = self._extend(kernel_size, num_layers)
        hidden_dim = self._extend(hidden_dim, num_layers)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(cur_input_dim, self.hidden_dim[i], self.kernel_size[i], self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.transpose(0, 1)
        b, _, _, h, w = input_tensor.size()
        if hidden_state is None:
            hidden_state = [self.cell_list[i].init_hidden(b, (h, w)) for i in range(self.num_layers)]

        layer_output_list = []
        last_state_list = []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h_s, c_s = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h_s, c_s = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], [h_s, c_s])
                output_inner.append(h_s)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h_s, c_s])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        return layer_output_list, last_state_list

    @staticmethod
    def _extend(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


# ── IMTFE (Image Manipulation Trace Feature Extractor) ─────────────────────────

class IMTFE(nn.Module):
    def __init__(self, in_channel=3, device=None):
        super().__init__()
        if device is None:
            device = torch.device('cpu')
        self.relu = nn.ReLU()
        self.device = device

        self.init_conv = nn.Conv2d(in_channel, 4, 5, 1, padding=0, bias=False)
        self.BayarConv2D = nn.Conv2d(in_channel, 3, 5, 1, padding=0, bias=False)
        self.bayar_mask = torch.tensor(np.ones(shape=(5, 5))).to(self.device)
        self.bayar_mask[2, 2] = 0
        self.bayar_final = torch.tensor(np.zeros((5, 5))).to(self.device)
        self.bayar_final[2, 2] = -1

        self.SRMConv2D = nn.Conv2d(in_channel, 9, 5, 1, padding=0, bias=False)
        # SRM weights will be loaded from checkpoint; freeze them
        for param in self.SRMConv2D.parameters():
            param.requires_grad = False

        self.middle_and_last_block = nn.ModuleList([
            nn.Conv2d(16, 32, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=0),
        ])

    def forward(self, x):
        # Normalize to [-1, 1]
        x = x / 255.0 * 2 - 1

        # Bayar constraint enforcement
        self.BayarConv2D.weight.data *= self.bayar_mask
        self.BayarConv2D.weight.data *= torch.pow(
            self.BayarConv2D.weight.data.sum(axis=(2, 3)).view(3, 3, 1, 1), -1
        )
        self.BayarConv2D.weight.data += self.bayar_final

        # Symmetric padding
        x = symm_pad(x, (2, 2, 2, 2))

        conv_init = self.init_conv(x)
        conv_bayar = self.BayarConv2D(x)
        conv_srm = self.SRMConv2D(x)

        first_block = torch.cat([conv_init, conv_srm, conv_bayar], axis=1)
        first_block = self.relu(first_block)

        last_block = first_block
        for layer in self.middle_and_last_block:
            if isinstance(layer, nn.Conv2d):
                last_block = symm_pad(last_block, (1, 1, 1, 1))
            last_block = layer(last_block)

        # L2 normalization
        last_block = F.normalize(last_block, dim=1, p=2)
        return last_block


# ── Anomaly Detector ───────────────────────────────────────────────────────────

class AnomalyDetector(nn.Module):
    def __init__(self, eps=1e-6, device=None):
        super().__init__()
        if device is None:
            device = torch.device('cpu')
        self.eps = eps
        self.relu = nn.ReLU()
        self.device = device

        self.adaptation = nn.Conv2d(256, 64, 1, 1, padding=0, bias=False)
        self.sigma_F = nn.Parameter(torch.zeros((1, 64, 1, 1)), requires_grad=True)

        self.pool31 = nn.AvgPool2d(31, stride=1, padding=15, count_include_pad=False)
        self.pool15 = nn.AvgPool2d(15, stride=1, padding=7, count_include_pad=False)
        self.pool7 = nn.AvgPool2d(7, stride=1, padding=3, count_include_pad=False)

        self.conv_lstm = ConvLSTM(
            input_dim=64, hidden_dim=8, kernel_size=(7, 7),
            num_layers=1, batch_first=False, bias=True, return_all_layers=False,
        )
        self.end = nn.Sequential(nn.Conv2d(8, 1, 7, 1, padding=3), nn.Sigmoid())

    def forward(self, IMTFE_output):
        _, _, H, W = IMTFE_output.shape
        self.GlobalPool = nn.AvgPool2d((H, W), stride=1)

        X_adapt = self.adaptation(IMTFE_output)
        X_adapt = batch_norm(X_adapt)

        # Multi-scale Z-pooling
        mu_T = self.GlobalPool(X_adapt)
        sigma_T = torch.sqrt(self.GlobalPool(torch.square(X_adapt - mu_T)))
        sigma_T = torch.max(sigma_T, self.sigma_F + self.eps)
        inv_sigma_T = torch.pow(sigma_T, -1)

        zpoolglobal = torch.abs((mu_T - X_adapt) * inv_sigma_T)
        zpool31 = torch.abs((self.pool31(X_adapt) - X_adapt) * inv_sigma_T)
        zpool15 = torch.abs((self.pool15(X_adapt) - X_adapt) * inv_sigma_T)
        zpool7 = torch.abs((self.pool7(X_adapt) - X_adapt) * inv_sigma_T)

        input_rnn = torch.cat([
            zpool7.unsqueeze(0), zpool15.unsqueeze(0),
            zpool31.unsqueeze(0), zpoolglobal.unsqueeze(0),
        ], axis=0)

        _, output_lstm = self.conv_lstm(input_rnn)
        output_lstm = output_lstm[0][0]
        return self.end(output_lstm)


# ── ManTraNet (Full Model) ─────────────────────────────────────────────────────

class MantraNet(nn.Module):
    """
    ManTra-Net: Manipulation Tracing Network for Detection and
    Localization of Image Forgeries With Anomalous Features (CVPR 2019)

    Input:  RGB image tensor (B, 3, H, W) — raw pixel values [0, 255]
    Output: Forgery probability map (B, 1, H, W) — values [0, 1]
    """

    def __init__(self, in_channel=3, eps=1e-6, device=None):
        super().__init__()
        if device is None:
            device = torch.device('cpu')
        self.eps = eps
        self.device = device
        self.IMTFE = IMTFE(in_channel=in_channel, device=device)
        self.AnomalyDetector = AnomalyDetector(eps=eps, device=device)

    def forward(self, x):
        return self.AnomalyDetector(self.IMTFE(x))


def load_pretrained_mantranet(weight_path, device=None):
    """Load ManTraNet with pre-trained weights."""
    if device is None:
        device = torch.device('cpu')
    model = MantraNet(device=device)
    state_dict = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
