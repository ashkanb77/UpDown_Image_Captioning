from torch import nn
import torch


class GatedTanh(nn.Module):
    def __init__(self, inp_size, out_size):
        super(GatedTanh, self).__init__()
        self.l1 = nn.Linear(inp_size, out_size)
        self.l2 = nn.Linear(inp_size, out_size)

    def forward(self, X):
        y = torch.tanh(self.l1(X))
        g = torch.sigmoid(self.l2(X))
        return y * g


class UpDown(nn.Module):
    def __init__(self, embed_size, hidden_size, img_vec_size, label_size, token_size, pad_index):
        super(UpDown, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.img_vec_size = img_vec_size
        self.label_size = label_size
        self.token_size = token_size
        self.pad_index = pad_index
        self.embedding = nn.Embedding(token_size, embed_size, pad_index)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.g1 = GatedTanh(hidden_size + img_vec_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, 1)
        self.soft = nn.Softmax(dim=1)
        self.g2 = GatedTanh(img_vec_size, hidden_size)
        self.g3 = GatedTanh(hidden_size, hidden_size)
        self.g4 = GatedTanh(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, label_size)

    def forward(self, V, Q):
        Q = self.embedding(Q)
        o, h = self.gru(Q)
        h = h[0]

        Q = h.unsqueeze(1).repeat(1, V.shape[1], 1)
        X = torch.concat((V, Q), 2)
        X = self.g1(X)
        X = self.l1(X)
        X = self.soft(X)

        V = V * X
        V = V.sum(dim=1)
        V = self.g2(V)

        h = self.g3(h)

        X = h * V
        X = self.g4(X)
        X = self.l2(X)

        return X

