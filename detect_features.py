import torch
from config import W, H, device
import torch.nn as nn


def detect_features_faster_rcnn(faster_rcnn, batch_images, n_features):
    faster_rcnn.eval()
    features = torch.zeros(batch_images.shape[0], n_features, 256, device=device)

    for img_idx, img in enumerate(batch_images):

        o = faster_rcnn(img.view(1, 3, W, H))
        boxes = o[0]['boxes']
        b = faster_rcnn.backbone(img.view(1, 3, W, H))
        b = b['3']

        boxes_ids = torch.div(boxes, 32, rounding_mode='trunc')
        boxes_ids = boxes_ids.type(torch.uint8)
        boxes_ids = torch.unique(boxes_ids, dim=0)

        z = torch.zeros((n_features, 256))
        avg11 = nn.AdaptiveAvgPool2d(1)

        for i in range(min(boxes_ids.shape[0], z.shape[0])):
            z[i, :] = torch.flatten(
                avg11(
                    b[:, :, boxes_ids[i, 1]:boxes_ids[i, 3] + 1, boxes_ids[i, 0]:boxes_ids[i, 2] + 1]
                )[0]
            )

        features[img_idx, :, :] = z[:, :]

    return features


def detect_features_cnn(net, batch_images, n_features):
    avg = nn.AdaptiveAvgPool2d(n_features // 2)
    with torch.no_grad():
        features = net(batch_images).to(device)
        features = avg(features)
        features = features.permute((0, 2, 3, 1))
        features = features.reshape((n_features, -1))

    return features
