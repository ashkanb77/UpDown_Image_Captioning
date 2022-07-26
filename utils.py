import re
import os
import torch
from config import device
from model import UpDown


def clean(s):
  try:
     return ' '.join(re.findall(r'[a-z0-9]+', s))
  except:
    return s


def process_caption(annots):
    d = dict()

    for row in annots:

        for word in clean(row['caption']).split(' '):
            count = d.get(word.lower())
            if count == None:
                d.update({word.lower(): 1})
            else:
                d[word.lower()] += 1

    tokens = set()
    for key in d.keys():
        if d[key] > 5:
            tokens.add(key)

    captions = []
    max_caption_len = 0
    for row in annots:
        s = []
        for word in clean(row['caption'].lower()).split(' '):
            count = d.get(word)
            if count is not None and count > 5:
                s.append(word)

        captions.append((row['image_id'], ' '.join(s)))
        max_caption_len = max(max_caption_len, len(s))

    tokens = ['<PAD>', '<START>', '<END>'] + list(tokens)

    print(
        f"number of captions: {len(captions)} " + \
        f"and number fo tokens: {len(tokens)} " + \
        f"and max length of captions: {max_caption_len}"
    )

    return captions, max_caption_len, tokens


class Checkpoint:

    def __init__(self, model, optimizer, file_name, dir_path):
        self.best_loss = 1000
        self.folder = dir_path
        self.model = model
        self.optimizer = optimizer
        self.file_name = file_name
        os.makedirs(self.folder, exist_ok=True)

    def save(self, loss, feature_extractor, lr=0.001, epoch=-1):
        if loss < self.best_loss:
            state = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'loss': loss,
                'learning_rate': lr,
                'epoch': epoch,
                'embed_size': self.model.embed_size,
                'h1_size': self.model.h1_size,
                'h2_size': self.model.h2_size,
                'img_vec_size': self.model.img_vec_size,
                'mid_lin': self.model.mid_lin,
                'token_size': self.model.token_size,
                'pad_index': self.model.pad_index,
                'feature_extractor': feature_extractor
            }
            path = os.path.join(os.path.abspath(self.folder), self.file_name + '.pth')
            torch.save(state, path)
            self.best_loss = loss

    @staticmethod
    def load_model(path):
        checkpoint = torch.load(path, map_location=device)
        model = UpDown(
            checkpoint['embed_size'], checkpoint['h1_size'], checkpoint['h2_size'],
            checkpoint['img_vec_size'], checkpoint['mid_lin'], checkpoint['token_size'], checkpoint['pad_index']
        )
        optimizer = torch.optim.Adam(model.parameters(), checkpoint['learning_rate'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        return model, optimizer, checkpoint['loss'], checkpoint['feature_extractor']
