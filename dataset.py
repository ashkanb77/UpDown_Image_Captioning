import torch
from torch.utils.data import Dataset
from PIL import Image
from config import W, H
from utils import clean


class VQADataset(Dataset):
    def __init__(self, dir, annots, questions, labels, tokens, transforms, img_name):
        self.annots = annots
        self.questions = questions
        self.labels = labels
        self.tokens = tokens
        self.transforms = transforms
        self.dir = dir
        self.img_name = img_name

    def __getitem__(self, index):
        img_id = str(self.annots[index]['image_id'])
        img_name = '0' * (12 - len(img_id))
        img_name += img_id
        img_name = self.img_name + img_name + '.jpg'
        img = Image.open(self.dir + img_name)
        if img.mode == 'L' or img.mode == 'LA':
            rgbimg = Image.new("RGB", img.size)
            rgbimg.paste(img)
            img = rgbimg.copy()
        img = self.transforms(img)

        question = self.questions[index]['question']
        question = clean(question)
        question = [self.tokens.index(w) for w in question.split(' ')]

        if len(question) > 14:
            question = question[0:14]
        elif len(question) < 14:
            question += [self.tokens.index('<PAD>') for i in range(14 - len(question))]

        label = self.labels.index(self.annots[index]['multiple_choice_answer'])

        return {
            'images': img,
            'questions': torch.tensor(question),
            'labels': label
        }

    def __len__(self):
        return len(self.annots)