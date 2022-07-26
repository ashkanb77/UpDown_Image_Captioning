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
parser.add_argument('--question', type=str, required=True, help='input question')

args = parser.parse_args()

model, optimizer, loss, feature_extractor = Checkpoint.load_model(args.checkpoint_path)

if feature_extractor == 'faster_rcnn':
    faster_rcnn = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    V_SIZE = 256

elif feature_extractor == 'resnet':
    net = resnet50(weights=ResNet50_Weights.DEFAULT)
    net = nn.Sequential(*(list(net.children())[:-2])).to(device)
else:
    net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    net = nn.Sequential(*(list(net.children())[:-2])).to(device)

img = Image.open(args.image_path)

with open('meta_data/tokens.pkl', 'rb') as file:
    tokens = pickle.load(file)


question = [tokens.index(w) for w in args.question.split(' ')]

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

    questions = torch.tensor(question).view(1, -1).to(device)

    outputs = model(features, questions)

    print(tokens[(outputs.softmax(dim=-1).argmax(axis=1).cpu().numpy()[0])])
