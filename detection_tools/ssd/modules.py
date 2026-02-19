from typing import Union, Optional

from einops import rearrange
import torch
from torch import nn
from torchvision import models
from torchvision.tv_tensors import BoundingBoxes

from detection_tools.ssd.utils import OffsetHandler


def _init_xavier(module):
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class VGG16FeatureExtractor(nn.Module):
    """This class extracts features from the input image using VGG16 architecture."""

    MAP_SHAPES_300 = [
        (38, 38),  # conv4_3
        (19, 19),  # fc7
        (10, 10),  # conv8
        (5, 5),  # conv9
        (3, 3),  # conv10
        (1, 1),  # conv11
    ]
    MAP_CHANNELS = [512, 1024, 512, 256, 256, 256]

    def __init__(self, freeze_till: int = None):
        super().__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        features = vgg16.features

        # Turning the pooliing layers to ceil mode to avoid mismatch in feature map sizes
        pool_pos = []
        for i, layer in enumerate(features):
            if freeze_till is not None and i <= freeze_till:
                layer.requires_grad_(False)
            if isinstance(layer, nn.MaxPool2d):
                pool_pos.append(i)
                layer.ceil_mode = True
        # As per SSD paper, we need to extract features from conv4_3
        self.features_conv4_3 = nn.Sequential(*features[: pool_pos[3]])

        # Scaling parameters for conv4_3 features as per SSD paper
        self.conv4_3_scaling = nn.Parameter(torch.ones(512) * 20.0)

        self.mod_pool5 = nn.MaxPool2d(
            kernel_size=3, stride=1, padding=1, ceil_mode=True
        )
        # Getting remaining features back
        self.remaining_features = nn.Sequential(
            *features[pool_pos[3] : pool_pos[4]], self.mod_pool5
        )
        # Adding extra layers to extract features as per the SSD paper
        self.fc6 = nn.Sequential(
            nn.Conv2d(
                512, 1024, kernel_size=3, padding=6, dilation=6
            ),  # Atrous convolution
            nn.ReLU(inplace=True),
        )
        _init_xavier(self.fc6)
        self.fc7 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1), nn.ReLU(inplace=True)
        )
        _init_xavier(self.fc7)
        # Adding additional convolutional layers to extract multiscale features
        self.conv8 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        _init_xavier(self.conv8)
        self.conv9 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        _init_xavier(self.conv9)
        self.conv10 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        _init_xavier(self.conv10)
        self.conv11 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        _init_xavier(self.conv11)

    def forward(self, X: torch.Tensor) -> list[torch.Tensor]:
        conv4_3_features = self.features_conv4_3(X)
        conv4_3_features = nn.functional.normalize(
            conv4_3_features
        ) * self.conv4_3_scaling.view(1, -1, 1, 1)
        pool5_features = self.remaining_features(conv4_3_features)
        fc6_features = self.fc6(pool5_features)
        fc7_features = self.fc7(fc6_features)
        conv8_features = self.conv8(fc7_features)
        conv9_features = self.conv9(conv8_features)
        conv10_features = self.conv10(conv9_features)
        conv11_features = self.conv11(conv10_features)
        return [
            conv4_3_features,
            fc7_features,
            conv8_features,
            conv9_features,
            conv10_features,
            conv11_features,
        ]


class SSDLoss(nn.Module):
    """This class computes the loss for SSD model."""

    def __init__(
        self,
        neg_pos_ratio: int = 3,
        alpha: float = 1.0,
        o_handler: Optional[OffsetHandler] = None,
    ):
        super().__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.cls_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.reg_loss_fn = nn.SmoothL1Loss(reduction="none")
        self.o_handler = o_handler
        if o_handler is None:
            self.o_handler = OffsetHandler()

    def forward(
        self,
        targets: list[dict[str, torch.Tensor]],
        head_outputs: dict[str, torch.Tensor],
        anchors: list[BoundingBoxes],
        matches: list[torch.Tensor],
        raw: bool = False,
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        pred_offsets = head_outputs["offsets"]
        pred_cls_logits = head_outputs["cls_logits"]

        actual_matched_anchor_classes = []
        actual_matched_gt_boxes = []
        actual_matched_anchors = []
        pred_foreground_offsets = []
        for anchor, match, target, pred_offset in zip(
            anchors, matches, targets, pred_offsets
        ):
            # Collecting the matching boxes
            foreground_anchor_idxs = torch.where(match >= 0)[0]
            foreground_gt_box_idxs = match[foreground_anchor_idxs]
            foreground_match_gt_boxes = target["bbox"][foreground_gt_box_idxs]
            actual_matched_gt_boxes.append(foreground_match_gt_boxes)
            actual_matched_anchors.append(anchor[foreground_anchor_idxs])
            # Collecting the matching offsets
            pred_foreground_offsets.append(pred_offset[foreground_anchor_idxs])
            # Collecting the matching classes
            anchor_classes = torch.zeros_like(match)  # Background class is 0
            anchor_classes[foreground_anchor_idxs] = target["labels"][
                foreground_gt_box_idxs
            ]
            actual_matched_anchor_classes.append(anchor_classes)
        # Computing the regression offset loss
        actual_offsets = self.o_handler.generate_offsets(
            actual_matched_gt_boxes, actual_matched_anchors
        )
        actual_offsets = torch.cat(actual_offsets, dim=0)
        pred_foreground_offsets = torch.cat(pred_foreground_offsets, dim=0)
        reg_loss = self.reg_loss_fn(pred_foreground_offsets, actual_offsets).sum()
        # Computing the classification loss with hard negative mining
        actual_matched_anchor_classes = torch.stack(
            actual_matched_anchor_classes, dim=0
        )
        cls_loss = self.cls_loss_fn(
            pred_cls_logits.view(-1, pred_cls_logits.shape[-1]),
            actual_matched_anchor_classes.view(-1),
        ).view(actual_matched_anchor_classes.size())
        # Performing hard negative mining
        # Hard negatives are those negatives which are hard to discern from positives
        # meaning their loss value will be highest among all negatives
        pos_mask = actual_matched_anchor_classes > 0
        num_pos = pos_mask.sum(dim=1, keepdim=True)
        num_neg = self.neg_pos_ratio * num_pos
        cls_loss_clone = cls_loss.clone()
        cls_loss_clone[pos_mask] = float(
            "-inf"
        )  # So that they are not selected as hard negatives
        # Obtaining a mapping for each anchor set: sort_order -> anchor_ids
        negative_idxs = cls_loss_clone.argsort(dim=1, descending=True)
        # Obtaining a mapping: anchor_ids -> sort_order
        ranks = negative_idxs.argsort(dim=1)
        hard_neg_mask = ranks < num_neg
        hard_neg_mask = (~pos_mask) & hard_neg_mask
        # Computing the final losses
        num_pos_total = max(1, num_pos.sum())  # To avoid division by zero
        final_cls_loss = (
            cls_loss[pos_mask].sum() + cls_loss[hard_neg_mask].sum()
        ) / num_pos_total
        final_reg_loss = reg_loss / num_pos_total
        if raw:
            return {"reg_loss": final_reg_loss, "cls_loss": final_cls_loss}
        return self.alpha * reg_loss + final_cls_loss


class SSDScoreHead(nn.Module):
    """Defines a general SSD head."""

    def __init__(self, num_scorers: int, num_anchors: list[int], channels: list[int]):
        super().__init__()
        assert len(num_anchors) == len(
            channels
        ), "Length of num_anchors and channels must match."
        self.num_scorers = num_scorers
        self.num_anchors = num_anchors
        self.channels = channels
        self.scorers = nn.ModuleList()
        for num_anchor, in_channels in zip(num_anchors, channels):
            scorer = nn.Conv2d(
                in_channels, num_anchor * num_scorers, kernel_size=3, padding=1
            )
            self.scorers.append(scorer)
        self.apply(_init_xavier)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        scores = []
        for i, feature in enumerate(features):
            score = self.scorers[i](feature)
            score = rearrange(
                score, "b (a c) h w -> b (h w a) c", a=self.num_anchors[i]
            )
            scores.append(score)
        return torch.cat(scores, dim=1)


class SSDHead(nn.Module):
    """Defines the SSD head which consists of a score head and an offset head."""

    def __init__(
        self,
        num_classes: int,
        num_anchors: list[int],
        channels: list[int],
    ):
        super().__init__()
        self.offset_regressor = SSDScoreHead(
            num_scorers=4, num_anchors=num_anchors, channels=channels
        )
        self.classifier = SSDScoreHead(
            num_scorers=num_classes, num_anchors=num_anchors, channels=channels
        )

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        offsets = self.offset_regressor(features)
        cls_logits = self.classifier(features)
        return {"offsets": offsets, "cls_logits": cls_logits}
