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
from dataset import VQADataset
from detect_features import detect_features_cnn, detect_features_faster_rcnn
from model import UpDown
import pickle


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
parser.add_argument('--hidden_size', type=int, default=HIDDEN_SIZE, help='GRU hidden size')
parser.add_argument('--experiment', type=str, default='experiment1', help='experiment path')
parser.add_argument('--use_only_train2014', type=bool, default=False,
                    help='if limitation on hardware then use only train2014 dataset and split to train and test')
parser.add_argument('--feature_extractor', type=str, default='mobilenet',
                        choices=['mobilenet', 'resnet', 'faster_rcnn'])
parser.add_argument('--checkpoint_dir', type=str, default=CHECK_DIR, help='dataset directory')
parser.add_argument('--model_name', type=str, default=MODEL_NAME, help='dataset directory')

args = parser.parse_args()

if args.feature_extractor == 'faster_rcnn':
    faster_rcnn = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    V_SIZE = 256

elif args.feature_extractor == 'resnet':
    net = resnet50(weights=ResNet50_Weights.DEFAULT)
    net = nn.Sequential(*(list(net.children())[:-2])).to(device)
    V_SIZE = 2048
else:
    net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    net = nn.Sequential(*(list(net.children())[:-2])).to(device)
    V_SIZE = 1280


transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize([W, H]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomAffine(degrees=(-15, 15), translate=(0, 0.1), scale=(0.7, 1)),
        torchvision.transforms.RandomCrop(size=(W - 30, H - 30)),
        torchvision.transforms.Normalize(IMAGE_COLOR_MEAN, IMAGE_COLOR_STD),
    ])


if args.use_only_train2014:
    file = open(os.path.join(args.dataset_dir, 'google-train-annotations.json'), 'r')
    train_annots = json.load(file)
    file.close()

    file = open(os.path.join(args.dataset_dir, 'google-train.json'), 'r')
    train_questions = json.load(file)
    file.close()

    file = open(os.path.join(args.dataset_dir, 'google-val-annotations.json'), 'r')
    val_annots = json.load(file)
    file.close()

    file = open(os.path.join(args.dataset_dir, 'google-val.json'), 'r')
    val_questions = json.load(file)
    file.close()

    seperator_idx = len(train_annots)

    annots_list, questions_list, labels, tokens = process_qa(
        train_annots + val_annots, train_questions + val_questions, args.n_ans
    )

    train_annots = annots_list[:seperator_idx]
    val_annots = annots_list[seperator_idx:]

    train_questions = questions_list[:seperator_idx]
    val_questions = questions_list[seperator_idx:]

    train_images_dir = os.path.join(args.dataset_dir, 'train2014')
    val_images_dir = os.path.join(args.dataset_dir, 'val2014')

else:
    file = open(os.path.join(args.dataset_dir, 'google-train-annotations.json'), 'r')
    train_annots = json.load(file)
    file.close()

    file = open(os.path.join(args.dataset_dir, 'google-train.json'), 'r')
    train_questions = json.load(file)
    file.close()

    annots_list, questions_list, labels, tokens = process_qa(
        train_annots, train_questions, args.n_ans
    )

    train_annots, val_annots, train_questions, val_questions = train_test_split(
        annots_list, questions_list, test_size=0.1
    )

    train_images_dir = os.path.join(args.dataset_dir, 'train2014')
    val_images_dir = os.path.join(args.dataset_dir, 'train2014')

train_dataset = VQADataset(
    train_images_dir, train_annots, train_questions,
    labels, tokens, transforms, 'COCO_train2014_'
)

val_dataset = VQADataset(
    val_images_dir, val_annots, val_questions,
    labels, tokens, transforms, 'COCO_train2014_'
)

train_dataloader = DataLoader(train_dataset, BATCH_SIZE, True)
val_dataloader = DataLoader(val_dataset, BATCH_SIZE, True)


def train(model, train_dataloader, val_dataloader, checkpoint, optimizer, epochs, lr, plot=True):
    criterion = nn.CrossEntropyLoss()

    losses = []
    accs = []

    val_losses = []
    val_accs = []

    model.train()


    net.eval()

    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        n_batches = len(train_dataloader)

        model.train()
        with tqdm(train_dataloader, unit="batch") as tepoch:

            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")

                images = batch['images'].to(device)
                questions = batch['questions'].to(device)
                targets = batch['labels'].to(device)

                if args.feature_extractor == 'faster_rcnn':
                    features = detect_features_faster_rcnn(faster_rcnn, images, args.n_features)
                else:
                    features = detect_features_cnn(net, images, args.n_features)

                optimizer.zero_grad()
                outputs = model(features, questions)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                acc = (outputs.softmax(dim=-1).argmax(axis=1) == targets).sum() / BATCH_SIZE
                total_acc += acc

                tepoch.set_postfix(loss=loss.item(), accuracy=acc)

        total_loss = total_loss / n_batches
        total_acc = total_acc / n_batches

        losses.append(total_loss)
        accs.append(total_acc.cpu())

        val_loss, val_acc = eval(model, val_dataloader, checkpoint, epoch, lr)

        val_accs.append(val_acc.cpu())
        val_losses.append(val_loss)

        logger.info(
            f"Epoch: {epoch + 1}, Train Loss: {total_loss:.4}, Train Accuracy: {total_acc:.4}" \
            + f" Val Loss: {val_loss: .4}, Val Accuracy: {val_acc:.4}"
        )

    if plot:
        plt.title('Loss')
        plt.plot(losses, label='Train Loss')
        plt.plot(val_losses, label='Test Loss')
        plt.legend(loc='best')
        plt.show()

        plt.title('Accuracy')
        plt.plot(accs, label='Train Accuracy')
        plt.plot(val_accs, label='Test Accuracy')
        plt.legend(loc='best')
        plt.show()
    return losses, val_losses, accs, val_accs


def eval(model, val_dataloader, checkpoint, epoch, lr):
    criterion = nn.CrossEntropyLoss()

    model.eval()
    net.eval()

    total_loss = 0
    total_acc = 0
    n_batches = len(val_dataloader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            images = batch['images'].to(device)
            questions = batch['questions'].to(device)
            targets = batch['labels'].to(device)

            if args.feature_extractor == 'faster_rcnn':
                features = detect_features_faster_rcnn(faster_rcnn, images, args.n_features)
            else:
                features = detect_features_cnn(net, images, args.n_features)

            outputs = model(features, questions)

            loss = criterion(outputs, targets)
            total_loss += loss.item()
            acc = (outputs.softmax(dim=-1).argmax(axis=1) == targets).sum() / BATCH_SIZE
            total_acc += acc

    total_loss = total_loss / n_batches
    total_acc = total_acc / n_batches

    checkpoint.save(
        total_loss, model.embed_size, model.hidden_size, model.img_vec_size, model.label_size,
        model.token_size, model.pad_index, args.feature_extractor, lr=lr, epoch=epoch
    )

    return total_loss, total_acc


os.mkdir('meta_data')
with open('meta_data/tokens.pkl', 'wb') as file:
    pickle.dump(tokens, file)

model = UpDown(
    args.embed_size, args.hidden_size, V_SIZE,
    len(labels), len(tokens), tokens.index('<PAD>')
).to(device)
optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
checkpoint = Checkpoint(model, optimizer, args.model_name, args.checkpoint_dir)
