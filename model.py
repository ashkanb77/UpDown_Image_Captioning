from torch import nn
import torch
from config import device


class UpDown(nn.Module):
    def __init__(self, embed_size, h1, h2, img_vec_size, mid_lin, token_size, pad_index):
        super(UpDown, self).__init__()
        self.embed_size = embed_size
        self.img_vec_size = img_vec_size
        self.mid_lin = mid_lin
        self.token_size = token_size
        self.embedding = nn.Embedding(token_size, embed_size, pad_index)
        self.lstm1 = nn.LSTM(h2+img_vec_size+embed_size, h1)
        self.lstm2 = nn.LSTM(h1+img_vec_size, h2)
        self.lv = nn.Linear(img_vec_size, mid_lin)
        self.lh = nn.Linear(h1, mid_lin)
        self.attend = nn.Linear(mid_lin, 1)
        self.linear = nn.Linear(h2, token_size)
        self.soft = nn.Softmax(dim=0)
        self.h1_size = h1
        self.h2_size = h2

    def forward(self, V, w, h1, c1, h2, c2):
        vbar = V.mean(dim=0, keepdim=True)
        w = self.embedding(w)
        o, (h1, c1) = self.lstm1(torch.concat((
            h2.view(1, -1), vbar, w
        ), dim=1).view(1, 1, -1),
            (h1, c1)
        )

        vt = self.lv(V)
        h1t = self.lh(h1[0])
        h1t = h1t.repeat((V.shape[0], 1))

        a = torch.tanh(vt + h1t)
        a = self.attend(a)
        a = self.soft(a)

        v = a * V
        v = v.sum(dim=0, keepdim=True)

        o, (h2, c2) = self.lstm2(torch.concat((
            v, h1[0]
        ), dim=1).view(1, 1, -1),
            (h2, c2)
        )

        o = self.linear(h2[0])

        return o, h1, c1, h2, c2

    def get_hidden1(self):
        return torch.zeros((1, 1, self.h1_size)).to(device)

    def get_hidden2(self):
        return torch.zeros((1, 1, self.h2_size)).to(device)


