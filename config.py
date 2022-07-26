import torch

W, H = 256, 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
EPOCHS = 8
LEARNING_RATE = 0.001
N_ANS = 8
N_FEATURES = 16
IMAGE_COLOR_MEAN = (0.485, 0.456, 0.406)
IMAGE_COLOR_STD = (0.229, 0.224, 0.225)
DATASET_DIR = 'dataset'
EMBED_SIZE = 300
HIDDEN_SIZE = 512
CHECK_DIR = 'checkpoints'
MODEL_NAME = 'Up-Down'
