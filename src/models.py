import torch
from torch import nn, Tensor
import math
from captcha_solver.data.scripts.generator_config import CHAR_LIST


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=(1, 1), padding=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class AttBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super(AttBlock, self).__init__()
        self.dim = dim
        self.heads = heads
        self.inner_dim = dim * heads
        self.scale = dim ** (-0.5)
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(p=dropout)
        self.q = nn.Linear(self.dim, self.inner_dim, bias=False)
        self.k = nn.Linear(self.dim, self.inner_dim, bias=False)
        self.v = nn.Linear(self.dim, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.dim, bias=False)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        # print(f'q={q.shape}, k={k.shape}, v={v.shape}')
        q_k = torch.matmul(q, k.transpose(-2, -1))
        # print(f'q_k={q_k.shape}')
        q_k = self.softmax(q_k) / self.scale
        # print(f'q_k softmax={q_k.shape}')
        q_k_v = torch.matmul(q_k, v)
        q_k_v = self.dropout(q_k_v)
        # print(f'q_k_v={q_k_v.shape}')
        return self.o(q_k_v)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class CaptchaSolverLSTM(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaSolverLSTM, self).__init__()
        self.num_chars = num_chars
        self.f_extractor = nn.Sequential(
            ConvBlock(3, 256, kernel_size=3, stride=(1, 1), padding=0),
            nn.MaxPool2d(2, 2),
            ConvBlock(256, 192, kernel_size=3, stride=(1, 1), padding=0),
            nn.MaxPool2d(2, 2),
            ConvBlock(192, 128, kernel_size=3, stride=(1, 1), padding=0),
            nn.MaxPool2d(2, 2),
        )
        self.linear = nn.Sequential(
            nn.Linear(896, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.norm_layer_1 = nn.LayerNorm(128)
        self.norm_layer_2 = nn.LayerNorm(128)
        self.lstm = nn.LSTM(128, 64, num_layers=2, bidirectional=True, batch_first=True, dropout=0.5)
        self.final = nn.Linear(128, self.num_chars)

    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        x = self.f_extractor(x)
        # print(x.shape)
        x = x.permute(0, 3, 1, 2).flatten(2, -1)
        # print(x.shape)
        x = self.linear(x)
        x = self.norm_layer_1(x)
        x, _ = self.lstm(x)
        x = self.norm_layer_2(x)
        x = self.final(x)
        return x.permute(1, 0, 2)


class CaptchaSolverAtt(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaSolverAtt, self).__init__()
        self.num_chars = num_chars
        self.f_extractor = nn.Sequential(
            ConvBlock(3, 256, kernel_size=3, stride=(1, 1), padding=0),
            nn.MaxPool2d(2, 2),
            ConvBlock(256, 192, kernel_size=3, stride=(1, 1), padding=0),
            nn.MaxPool2d(2, 2),
            ConvBlock(192, 128, kernel_size=3, stride=(1, 1), padding=0),
            nn.MaxPool2d(2, 2),
        )
        self.linear = nn.Sequential(
            nn.Linear(896, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.pos_encoder = PositionalEncoding(128, dropout=0.25, max_len=35)
        self.attention = AttBlock(128, 4, dropout=0.5)
        self.final = nn.Linear(128, self.num_chars)

    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        x = self.f_extractor(x)
        # print(x.shape)
        x = x.permute(0, 3, 1, 2).flatten(2, -1)
        # print(x.shape)
        x = self.linear(x)
        x = self.pos_encoder(x)
        x = self.attention(x)
        x = self.final(x)
        return x.permute(1, 0, 2)


""" TESTING """
if __name__ == "__main__":
    model = CaptchaSolverLSTM(num_chars=len(CHAR_LIST))
    inp = torch.rand(1, 3, 75, 200)
    out = model(inp)
    print(model.f_extractor[0].block[0].weight.grad)
    print(out.shape)
