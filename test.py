import torchvision
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models import resnet50, mobilenet_v3_small, ResNet50_Weights, MobileNet_V3_Small_Weights
import argparse
from PIL import Image
from utils import Checkpoint
from config import *
import pickle
from detect_features import detect_features_faster_rcnn, detect_features_cnn

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path', type=str, required=True, help='checkpoint path')
parser.add_argument('--image_path', type=str, required=True, help='input image path')

args = parser.parse_args()

model, optimizer, loss, feature_extractor = Checkpoint.load_model(args.checkpoint_path)

if feature_extractor == 'faster_rcnn':
    faster_rcnn = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
elif feature_extractor == 'resnet':
    net = resnet50(weights=ResNet50_Weights.DEFAULT)
    net = nn.Sequential(*(list(net.children())[:-2])).to(device)
else:
    net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    net = nn.Sequential(*(list(net.children())[:-2])).to(device)

img = Image.open(args.image_path)

with open('meta_data/tokens.pkl', 'rb') as file:
    tokens = pickle.load(file)

transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize([W, H]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGE_COLOR_MEAN, IMAGE_COLOR_STD),
    ])


with torch.no_grad():
    img = transforms(img).to(device)

    if feature_extractor == 'faster_rcnn':
        features = detect_features_faster_rcnn(faster_rcnn, img.view(1, 3, W, H), N_FEATURES)
    else:
        features = detect_features_cnn(net, img.view(1, 3, W, H), N_FEATURES)

    s = ['<START>']

    with torch.no_grad():
        h1, c1 = model.get_hidden1(), model.get_hidden1()
        h2, c2 = model.get_hidden2(), model.get_hidden2()
        for widx in range(300):
            o, h1, c1, h2, c2 = model(
                features, torch.tensor([tokens.index(s[-1])]).to(device), h1, c1, h2, c2
            )

            s.append(tokens[o.softmax(dim=1).argmax()])
            if s[-1] == '<END>':
                break

    print(' '.join(s[1:-1]))
