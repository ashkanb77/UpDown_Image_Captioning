import argparse
import logging
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models import resnet50, mobilenet_v3_small, ResNet50_Weights, MobileNet_V3_Small_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from config import *
from utils import *
from dataset import CaptioningDataset
from detect_features import detect_features_cnn, detect_features_faster_rcnn
from model import UpDown
import pickle
import json


logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger('Face-Model')

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=EPOCHS, help='number of epochs for training')
parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR, help='dataset directory')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size')
parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='learning rate')
parser.add_argument('--n_features', type=int, default=N_FEATURES, help='number of features of image to use')
parser.add_argument('--n_ans', type=int, default=N_ANS, help='min of answers should be apper in dataset')
parser.add_argument('--embed_size', type=int, default=EMBED_SIZE, help='word embedding size')
parser.add_argument('--h1_size', type=int, default=H1_SIZE, help='first lstm hidden size')
parser.add_argument('--h2_size', type=int, default=H2_SIZE, help='second lstm hidden size')
parser.add_argument('--mid_lin', type=int, default=MID_LINEAR, help='middle linear size in attend module')
parser.add_argument('--experiment', type=str, default='experiment1', help='experiment path')
parser.add_argument('--use_only_train2014', type=int, default=1, choices=[0, 1],
                    help='if limitation on hardware then use only train2014 dataset and split to train and test')
parser.add_argument('--feature_extractor', type=str, default='mobilenet',
                        choices=['mobilenet', 'resnet', 'faster_rcnn'])
parser.add_argument('--checkpoint_dir', type=str, default=CHECK_DIR, help='dataset directory')
parser.add_argument('--model_name', type=str, default=MODEL_NAME, help='dataset directory')

args = parser.parse_args()

if args.feature_extractor == 'faster_rcnn':
    faster_rcnn = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    faster_rcnn.eval()
    V_SIZE = 256

elif args.feature_extractor == 'resnet':
    net = resnet50(weights=ResNet50_Weights.DEFAULT)
    net = nn.Sequential(*(list(net.children())[:-2])).to(device)
    net.eval()
    V_SIZE = 2048
else:
    net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    net = nn.Sequential(*(list(net.children())[:-2])).to(device)
    net.eval()
    V_SIZE = 576


transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize([W, H]),
        torchvision.transforms.RandomCrop(size=(W - 30, H - 30)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomAffine(degrees=(-15, 15), translate=(0, 0.1), scale=(0.7, 1)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGE_COLOR_MEAN, IMAGE_COLOR_STD),
    ])


if  args.use_only_train2014 == 0:
    file = open('dataset/annotations/captions_train2014.json', 'r')
    train_annots = json.load(file)
    file.close()

    file = open('dataset/annotations/captions_val2014.json', 'r')
    val_annots = json.load(file)
    file.close()

    seperator_idx = len(train_annots['annotations'])

    captions, max_caption_len, tokens = process_caption(
        train_annots['annotations'] + val_annots['annotations']
    )

    train_captions = captions[:seperator_idx]
    val_captions = captions[seperator_idx:]

    train_images_dir = os.path.join(args.dataset_dir, 'train2014')
    val_images_dir = os.path.join(args.dataset_dir, 'val2014')

else:
    file = open('dataset/annotations/captions_train2014.json', 'r')
    train_annots = json.load(file)
    file.close()

    captions, max_caption_len, tokens = process_caption(
        train_annots['annotations']
    )

    train_captions, val_captions = train_test_split(
        captions, test_size=0.1
    )

    train_images_dir = os.path.join(args.dataset_dir, 'train2014')
    val_images_dir = os.path.join(args.dataset_dir, 'train2014')

train_dataset = CaptioningDataset(
    train_images_dir, train_captions, tokens,
    transforms, 'COCO_train2014_', max_caption_len
)


if args.use_only_train2014 == 0:
    val_dataset = CaptioningDataset(
        val_images_dir, val_captions, tokens,
        transforms, 'COCO_val2014_', max_caption_len
    )
else:
    val_dataset = CaptioningDataset(
        val_images_dir, val_captions, tokens,
        transforms, 'COCO_train2014_', max_caption_len
    )

train_dataloader = DataLoader(train_dataset, args.batch_size, True)
val_dataloader = DataLoader(val_dataset, args.batch_size, True)


def train(model, train_dataloader, val_dataloader, checkpoint, optimizer, epochs, lr, plot=True):
    criterion = nn.CrossEntropyLoss()

    losses = []
    val_losses = []

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        n_batches = len(train_dataloader)

        model.train()
        with tqdm(train_dataloader, unit="batch") as tepoch:

            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")

                images = batch['images'].to(device)
                captions = batch['captions'].to(device)

                if args.feature_extractor == 'faster_rcnn':
                    features = detect_features_faster_rcnn(faster_rcnn, images, args.n_features)
                else:
                    features = detect_features_cnn(net, images, args.n_features)

                batch_loss = 0
                for i, caption in enumerate(captions):
                    loss = 0
                    optimizer.zero_grad()

                    h1, c1 = model.get_hidden1(), model.get_hidden1()
                    h2, c2 = model.get_hidden2(), model.get_hidden2()
                    for widx in range(caption.shape[0] - 1):
                        o, h1, c1, h2, c2 = model(
                            features[i], caption[widx].view(1), h1, c1, h2, c2
                        )

                        loss += criterion(o, caption[widx + 1].view(1))
                        if tokens.index('<END>') == caption[widx + 1].cpu():
                            break

                    batch_loss = batch_loss + loss.item() / (widx + 1)
                    loss.backward()
                    optimizer.step()

                batch_loss = batch_loss / BATCH_SIZE
                total_loss = total_loss + batch_loss
                tepoch.set_postfix(loss=batch_loss)

            total_loss = total_loss / n_batches

            losses.append(total_loss)

            val_loss = eval(model, val_dataloader, checkpoint, epoch, lr)

            val_losses.append(val_loss)

            print(
                f"Epoch: {epoch + 1}, Train Loss: {total_loss:.4},"\
                + f" Val Loss: {val_loss: .4}"
            )

        if plot:
            plt.title('Loss')
            plt.plot(losses, label='Train Loss')
            plt.plot(val_losses, label='Test Loss')
            plt.legend(loc='best')
            plt.show()


def eval(model, val_dataloader, checkpoint, epoch, lr):
    criterion = nn.CrossEntropyLoss()

    model.eval()
    net.eval()

    total_loss = 0
    n_batches = len(val_dataloader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):

            images = batch['images'].to(device)
            captions = batch['captions'].to(device)

            if args.feature_extractor == 'faster_rcnn':
                features = detect_features_faster_rcnn(faster_rcnn, images, args.n_features)
            else:
                features = detect_features_cnn(net, images, args.n_features)

            batch_loss = 0
            for i, caption in enumerate(captions):
                loss = 0
                h1, c1 = model.get_hidden1(), model.get_hidden1()
                h2, c2 = model.get_hidden2(), model.get_hidden2()
                for widx in range(caption.shape[0] - 1):
                    o, h1, c1, h2, c2 = model(
                        features[i], caption[widx].view(1), h1, c1, h2, c2
                    )

                    loss += criterion(o, caption[widx + 1].view(1))
                    if tokens.index('<END>') == caption[widx + 1].cpu():
                        break
                loss = loss.item() / (widx + 1)
                batch_loss += loss

            total_loss = total_loss + batch_loss / args.batch_size

    total_loss = total_loss / n_batches
    checkpoint.save(total_loss, args.feature_extractor, lr=lr, epoch=epoch)

    return total_loss


os.makedirs('meta_data', exist_ok=True)
with open('meta_data/tokens.pkl', 'wb') as file:
    pickle.dump(tokens, file)

model = UpDown(
    args.embed_size, args.h1_size, args.h2_size, V_SIZE, args.mid_lin, len(tokens), tokens.index('<PAD>')
).to(device)
optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
checkpoint = Checkpoint(model, optimizer, args.model_name, args.checkpoint_dir)

train(
    model, train_dataloader, val_dataloader, checkpoint, optimizer, args.n_epochs, args.learning_rate
    )
