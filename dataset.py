import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class CaptioningDataset(Dataset):
    def __init__(self, dir, captions, tokens, transforms, img_name, max_len):
        self.captions = captions
        self.tokens = tokens
        self.transforms = transforms
        self.dir = dir
        self.img_name = img_name
        self.max_len = max_len

    def __getitem__(self, index):
        img_id = str(self.captions[index][0])
        img_name = '0' * (12 - len(img_id))
        img_name += img_id
        img_name = self.img_name + img_name + '.jpg'
        img = Image.open(os.path.join(self.dir, img_name))
        if img.mode == 'L' or img.mode == 'LA':
            rgbimg = Image.new("RGB", img.size)
            rgbimg.paste(img)
            img = rgbimg.copy()
        img = self.transforms(img)

        caption = self.captions[index][1]
        caption = '<START> ' + caption
        caption = [self.tokens.index(w) for w in caption.split(' ')]
        caption = caption + [self.tokens.index('<END>')]

        if len(caption) > self.max_len:
            caption = caption[0:self.max_len - 1] + [self.tokens.index('<END>')]
        elif len(caption) < self.max_len:
            caption += [self.tokens.index('<PAD>') for i in range(self.max_len - len(caption) - 1)]

        return {
            'images': img,
            'captions': torch.tensor(caption)
        }

    def __len__(self):
        return len(self.captions)
