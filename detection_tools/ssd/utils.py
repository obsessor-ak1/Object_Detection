import math
from typing import Optional, Union

import torch
from torchvision import ops
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes


class AnchorGenerator:
    """This class generates set of anchor boxes for a particular image."""

    def __init__(
        self,
        min_scale: float = 0.2,
        max_scale: float = 0.9,
        aspect_ratios: list[list[float]] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.aspect_ratios = aspect_ratios
        self.num_feature_maps = len(aspect_ratios)
        self.device = device
        self.scales = self._get_scales()
        self.height_width_pairs = self._get_height_width_pairs(
            self.scales, self.aspect_ratios
        )

    def _get_scales(self) -> torch.Tensor:
        """Returns a list of scales for each feature map."""
        scales = []
        for k in range(self.num_feature_maps):
            scale = self.min_scale + (self.max_scale - self.min_scale) * k / (
                self.num_feature_maps - 1
            )
            scales.append(scale)
        scales.append(1.0)
        return torch.tensor(scales, device=self.device)

    def _get_height_width_pairs(
        self, scales: list[float], aspect_ratios: list[list[float]]
    ) -> list[torch.Tensor]:
        """Returns height and width pairs for a particular scale and aspect ratios."""
        height_width_pairs = []
        for k in range(self.num_feature_maps):
            scale = scales[k]
            aspect_ratios_k = aspect_ratios[k]
            hw_k = []
            for aspect_ratio in aspect_ratios_k:
                height = scale / torch.sqrt(
                    torch.tensor(aspect_ratio, device=self.device)
                )
                width = scale * torch.sqrt(
                    torch.tensor(aspect_ratio, device=self.device)
                )
                hw_k.append((height, width))
                if aspect_ratio == 1.0:
                    s_prime = torch.sqrt(scale * scales[k + 1])
                    height_prime = s_prime / torch.sqrt(
                        torch.tensor(aspect_ratio, device=self.device)
                    )
                    width_prime = s_prime * torch.sqrt(
                        torch.tensor(aspect_ratio, device=self.device)
                    )
                    hw_k.append((height_prime, width_prime))
                else:
                    hw_k.append((width, height))
            height_width_pairs.append(torch.tensor(hw_k, device=self.device))
        return height_width_pairs

    def generate_anchors(
        self, image_size: tuple[int, int], feature_map_sizes: list[tuple[int, int]]
    ) -> BoundingBoxes:
        """Generates anchor boxes for a particular image and feature map sizes."""
        assert (
            len(feature_map_sizes) == self.num_feature_maps
        ), "Number of feature maps must match the number of aspect ratio sets."
        anchors = []
        img_height, img_width = image_size
        for k in range(self.num_feature_maps):
            feature_map_size = feature_map_sizes[k]
            height_width_pairs_k = self.height_width_pairs[k]
            anchors_k = self._generate_anchors_for_feature_map(
                feature_map_size, height_width_pairs_k
            )
            anchors.append(anchors_k)
        center_anchors = torch.cat(anchors, dim=0)
        center_abs_anchors = center_anchors * torch.tensor(
            [[img_width, img_height, img_width, img_height]], device=self.device
        )
        center_anchors = BoundingBoxes(
            center_abs_anchors,
            format="CXCYWH",
            device=self.device,
            canvas_size=image_size,
        )
        return v2.functional.convert_bounding_box_format(
            center_anchors, new_format="XYXY"
        )
    
    @property
    def num_anchors(self) -> list[int]:
        """Returns the number of anchors for each feature map per location."""
        return [len(wh_pair) for wh_pair in self.height_width_pairs]

    def _generate_anchors_for_feature_map(
        self, feature_map_size: tuple[int, int], height_width_pairs: torch.Tensor
    ) -> torch.Tensor:
        """Generates anchor boxes for a particular feature map."""
        fm_height, fm_width = feature_map_size
        y_ranges = (torch.arange(fm_height, device=self.device) + 0.5) / fm_height
        x_ranges = (torch.arange(fm_width, device=self.device) + 0.5) / fm_width
        center_y, center_x = torch.meshgrid(y_ranges, x_ranges, indexing="ij")
        center_x = center_x.flatten()
        center_y = center_y.flatten()
        # Each aspect_ratio/scale combination
        points = torch.stack([center_x, center_y], dim=1)
        points = points.repeat_interleave(height_width_pairs.shape[0], dim=0)
        height_width_pairs = height_width_pairs.repeat(fm_height * fm_width, 1)
        anchors = torch.cat([points, height_width_pairs], dim=1)
        return anchors


class OffsetHandler:
    """This class encodes and decodes bounding boxes using anchor boxes."""

    def __init__(
        self,
        weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        max_pred_box_size_offset: Optional[float] = math.log(1000.0 / 16.0)
    ):
        self.weights = weights
        self.max_pred_box_size_offset = max_pred_box_size_offset

    def generate_offsets(
        self, gt_boxes: list[BoundingBoxes], anchors: list[BoundingBoxes]
    ) -> list[torch.Tensor]:
        """Generates offsets for a batch of images."""
        assert len(gt_boxes) == len(
            anchors
        ), "Batch size of gt_boxes and anchors must match."
        num_boxes_per_image = [len(boxes) for boxes in gt_boxes]
        gt_boxes = torch.cat(gt_boxes, dim=0)
        anchors = torch.cat(anchors, dim=0)
        offsets = self.generate_offsets_single(gt_boxes, anchors)
        return offsets.split(num_boxes_per_image, dim=0)

    def generate_offsets_single(
        self, gt_boxes: Union[torch.Tensor, BoundingBoxes], anchors: Union[torch.Tensor, BoundingBoxes]
    ) -> torch.Tensor:
        if gt_boxes.numel() == 0:
            return torch.empty((0, 4), device=gt_boxes.device)
        wx, wy, ww, wh = self.weights
        gt_boxes = gt_boxes.as_subclass(torch.Tensor)
        anchors = anchors.as_subclass(torch.Tensor)
        gt_cxcywh = v2.functional.convert_bounding_box_format(
            gt_boxes, old_format="XYXY", new_format="CXCYWH"
        )
        anchors_cxcywh = v2.functional.convert_bounding_box_format(
            anchors, old_format="XYXY", new_format="CXCYWH"
        )
        dx = wx * (gt_cxcywh[..., 0] - anchors_cxcywh[..., 0]) / anchors_cxcywh[..., 2]
        dy = wy * (gt_cxcywh[..., 1] - anchors_cxcywh[..., 1]) / anchors_cxcywh[..., 3]
        dw = ww * torch.log((gt_cxcywh[..., 2] + 1e-6)/ anchors_cxcywh[..., 2])
        dh = wh * torch.log((gt_cxcywh[..., 3] + 1e-6)/ anchors_cxcywh[..., 3])
        offsets = torch.stack([dx, dy, dw, dh], dim=-1)
        return offsets

    def adjust_offset(
        self, offsets: list[torch.Tensor], anchors: list[BoundingBoxes]
    ) -> list[BoundingBoxes]:
        """Adjusts offsets for a batch of images."""
        assert len(offsets) == len(
            anchors
        ), "Batch size of offsets and anchors must match."
        num_boxes_per_image = [len(offset) for offset in offsets]
        canvas_sizes = [anchor.canvas_size for anchor in anchors]
        offsets = torch.cat(offsets, dim=0)
        anchors = torch.cat(anchors, dim=0)
        adjusted_boxes = self.adjust_offset_single(offsets, anchors)
        adjusted_boxes = adjusted_boxes.split(num_boxes_per_image, dim=0)
        # Converting adjusted boxes to BoundingBoxes format with appropriate canvas sizes
        adjusted_boxes = [
            BoundingBoxes(
                boxes, format="XYXY", device=anchors.device, canvas_size=canvas_size
            )
            for boxes, canvas_size in zip(adjusted_boxes, canvas_sizes)
        ]
        return adjusted_boxes

    def adjust_offset_single(
        self, offsets: torch.Tensor, anchors: Union[torch.Tensor, BoundingBoxes]
    ) -> BoundingBoxes:
        wx, wy, ww, wh = self.weights
        canvas_size = None
        if isinstance(anchors, BoundingBoxes):
            canvas_size = anchors.canvas_size
        anchors = anchors.as_subclass(torch.Tensor)
        anchors_cxcywh = v2.functional.convert_bounding_box_format(
            anchors, old_format="XYXY", new_format="CXCYWH"
        )
        cx = offsets[..., 0] / wx * anchors_cxcywh[..., 2] + anchors_cxcywh[..., 0]
        cy = offsets[..., 1] / wy * anchors_cxcywh[..., 3] + anchors_cxcywh[..., 1]
        dw = torch.clamp(offsets[..., 2] / ww, max=self.max_pred_box_size_offset)
        dh = torch.clamp(offsets[..., 3] / wh, max=self.max_pred_box_size_offset)
        w = torch.exp(dw) * anchors_cxcywh[..., 2]
        h = torch.exp(dh) * anchors_cxcywh[..., 3]
        adjusted_boxes = torch.stack([cx, cy, w, h], dim=-1)
        adjusted_boxes = BoundingBoxes(
            adjusted_boxes,
            format="CXCYWH",
            device=anchors.device,
            canvas_size=canvas_size
        )
        return v2.functional.convert_bounding_box_format(
            adjusted_boxes, new_format="XYXY"
        )


class Matcher:
    """Generates matches between ground truth boxes and anchor boxes using match
    quality matrix (Jaccard Index or IOU in this case)."""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes: BoundingBoxes, anchors: BoundingBoxes) -> torch.Tensor:
        iou_matrix = ops.box_iou(gt_boxes, anchors)
        max_iou, matches = iou_matrix.max(dim=0)
        # First assign the anchor to each gt box with which it has highest IOU and crosses the threshold
        matches[max_iou < self.iou_threshold] = -1
        # Now, making sure that each gt box must gets its best matching anchor box
        # even if that particular anchor box has max IOU with another gt box or doesn't cross the threshold
        _, best_anchor_for_gt = iou_matrix.max(dim=1)
        matches[best_anchor_for_gt] = torch.arange(
            len(gt_boxes), device=gt_boxes.device
        )
        return matches


class SSDPredictor:
    """This class predicts the final bounding boxes and classes from the SSD head
    outputs."""

    def __init__(
        self,
        score_thresh: float = 0.1,
        nms_thresh: float = 0.45,
        iou_thresh: float = 0.5,
        max_detections: int = 200,
        num_top_k: int = 100,
        device: torch.device = torch.device("cpu"),
        o_handler: Optional[OffsetHandler] = None,
    ):
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.iou_thresh = iou_thresh
        self.max_detections = max_detections
        self.num_top_k = num_top_k
        self.device = device
        self.o_handler = o_handler
        if o_handler is None:
            self.o_handler = OffsetHandler()

    def __call__(
        self,
        head_outputs: dict[str, torch.Tensor],
        anchors: list[BoundingBoxes],
        original_image_sizes: Optional[list[tuple[int, int]]] = None,
    ):
        pred_offsets = head_outputs["offsets"]
        pred_cls_scores = head_outputs["cls_logits"].softmax(dim=-1)

        num_classes = pred_cls_scores.size(-1)
        detections = []
        for i, (pred_offset, pred_cls_logit, anchor) in enumerate(
            zip(pred_offsets, pred_cls_scores, anchors)
        ):
            boxes = self.o_handler.adjust_offset_single(pred_offset, anchor)
            image_boxes = []
            image_labels = []
            image_scores = []
            # Now, we will collect top k boxes per class which have score above the threshold
            for cls_idx in range(1, num_classes):
                cls_scores = pred_cls_logit[:, cls_idx]
                # Filtering out boxes above the threshold
                above_thresh_idxs = torch.where(cls_scores > self.score_thresh)[0]
                if len(above_thresh_idxs) == 0:
                    continue
                cls_scores = cls_scores[above_thresh_idxs]
                cls_boxes = boxes[above_thresh_idxs]
                # Getting the top k scores and corresponding boxes
                top_k_picks = min(self.num_top_k, len(cls_scores))
                top_k_scores, top_k_idxs = torch.topk(cls_scores, k=top_k_picks)
                top_k_boxes = cls_boxes[top_k_idxs]
                image_boxes.append(top_k_boxes)
                image_scores.append(top_k_scores)
                image_labels.append(
                    torch.full_like(
                        top_k_scores, cls_idx, dtype=torch.long, device=self.device
                    )
                )
            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)
            # Now we will perform NMS to further filter the boxes
            # We will use batched NMS so that NMS is performed independently per class
            keep_idxs = ops.batched_nms(
                image_boxes, image_scores, image_labels, self.nms_thresh
            )
            # Keeping only top max_detections boxes after NMS
            keep_idxs = keep_idxs[: self.max_detections]
            final_boxes = image_boxes[keep_idxs]
            if original_image_sizes is not None:
                final_boxes = v2.functional.resize(final_boxes, size=original_image_sizes[i])
            detections.append(
                {
                    "bbox": final_boxes,
                    "labels": image_labels[keep_idxs],
                    "scores": image_scores[keep_idxs],
                }
            )
        return detections
