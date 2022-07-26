import re
import json
import os
import torch
from config import device
from model import UpDown


def clean(s):
  try:
     return ' '.join(re.findall(r'[آا-ی۱۲۳۴۵۶۷۸۹۰ئ]+', s))
  except:
    return s


def process_qa(annots, questions, n_ans):
    d = dict()

    for row in annots['annotations']:

        count = d.get(row['multiple_choice_answer'])
        if count == None:
            d.update({row['multiple_choice_answer']: 1})
        else:
            d[row['multiple_choice_answer']] += 1

    labels = []
    for key in d.keys():
        if d[key] > n_ans:
            labels.append(key)

    annots_list = []
    questions_list = []

    for i in range(len(annots['annotations'])):

        count = d.get(annots['annotations'][i]['multiple_choice_answer'])
        if count > n_ans:
            annots_list.append(annots['annotations'][i])
            questions_list.append(questions['questions'][i])

    tokens = set(d.keys()).copy()
    for row in questions['questions']:
        tokens.update(
            clean(row['question']).split(' ')
        )

    tokens = ['<PAD>'] + list(tokens)

    return annots_list, questions_list, labels, tokens


class Checkpoint:

    def __init__(self, model, optimizer, file_name, dir_path):
        self.best_loss = 1000
        self.folder = dir_path
        self.model = model
        self.optimizer = optimizer
        self.file_name = file_name
        os.makedirs(self.folder, exist_ok=True)

    def save(self, loss, embed_size, hidden_size, img_vec_size, label_size, token_size, pad_index,
             feature_extractor, lr=0.001, epoch=-1):

        if loss < self.best_loss:
            state = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'loss': loss,
                'learning_rate': lr,
                'epoch': epoch,
                'embed_size': embed_size,
                'hidden_size': hidden_size,
                'img_vec_size': img_vec_size,
                'label_size': label_size,
                'token_size': token_size,
                'pad_index': pad_index,
                'feature_extractor': feature_extractor
            }
            path = os.path.join(os.path.abspath(self.folder), self.file_name + '.pth')
            torch.save(state, path)
            self.best_loss = loss

    @staticmethod
    def load_model(path):
        checkpoint = torch.load(path, map_location=device)
        model = UpDown(
            checkpoint['embed_size'], checkpoint['hidden_size'], checkpoint['img_vec_size'],
            checkpoint['label_size'], checkpoint['token_size'], checkpoint['pad_index']
        )
        optimizer = torch.optim.Adam(model.parameters(), checkpoint['learning_rate'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        return model, optimizer, checkpoint['loss'], checkpoint['feature_extractor']
