import torch
from torchvision.datasets import VOCDetection
from torchvision import tv_tensors


VOC_CLASS_TO_IDX = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19
}
CLASSES = list(VOC_CLASS_TO_IDX.keys())

def tensorize_target(target):
    """Tensorize the target variable in the VOC Detection dataset
    provided by torchvision.datasets.VOCDetection."""
    objects = target["annotation"]["object"]
    height = float(target["annotation"]["size"]["height"])
    width = float(target["annotation"]["size"]["width"])

    bboxes = []
    labels = []
    for obj in objects:
        bbox = obj["bndbox"]
        xmin = float(bbox["xmin"])
        xmax = float(bbox["xmax"])
        ymin = float(bbox["ymin"])
        ymax = float(bbox["ymax"])
        name = obj["name"]
        label = VOC_CLASS_TO_IDX[name]
        labels.append(label)
        bboxes.append([xmin, ymin, xmax, ymax])
    
    if len(bboxes) > 0:
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        bboxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.long)

    return {
        "labels": labels,
        "bbox": tv_tensors.BoundingBoxes(
            bboxes, format="XYXY", canvas_size=(height, width),
            dtype=torch.float32
        )
    }


def remove_degenerate_boxes(target):
    """
    Removes boxes where:
        x_max <= x_min OR y_max <= y_min
    """
    boxes = target["bbox"]

    valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

    target["bbox"] = tv_tensors.BoundingBoxes(
        boxes[valid_mask],
        format=boxes.format,
        canvas_size=boxes.canvas_size
    )

    target["labels"] = target["labels"][valid_mask]

    return target


class SSDVOCDataset(VOCDetection):
    def __init__(self, background_present=True, transforms=None, **kwargs):
        super().__init__(transforms=None, **kwargs)
        self.actual_transforms = transforms
        self.background_present = background_present

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        # Convert annotation to tensor format
        target = tensorize_target(target)
        # Shift labels for background class
        if self.background_present:
            target["labels"] = target["labels"] + 1
        # Apply transforms (image + boxes together)
        if self.actual_transforms:
            img, target = self.actual_transforms(img, target)
        # Remove invalid boxes AFTER transforms
        target = remove_degenerate_boxes(target)
        return img, target