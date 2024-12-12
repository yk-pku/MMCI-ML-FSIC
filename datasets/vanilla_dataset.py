import PIL.Image as Image
import torch
import torchvision.transforms as transforms


train_cats = [
    'hair drier', 'clock', 'wine glass', 'book', 'cake', 'tie', 'motorcycle',
    'sheep', 'bottle', 'giraffe', 'cell phone', 'suitcase', 'remote', 'bench',
    'mouse', 'carrot', 'banana', 'train', 'sports ball', 'toothbrush', 'fire hydrant',
    'airplane', 'tv', 'bus', 'refrigerator', 'couch', 'knife', 'toilet', 'elephant',
    'truck', 'parking meter', 'car', 'potted plant', 'kite', 'skateboard', 'orange',
    'horse', 'cat', 'tennis racket', 'bowl', 'scissors', 'baseball glove', 'apple',
    'traffic light', 'handbag', 'donut', 'dog', 'hot dog', 'oven', 'umbrella', 'sink',
    'pizza'
]
val_cats = [
    'cow', 'dining table', 'zebra', 'sandwich', 'bear', 'toaster', 'person',
    'laptop', 'bed', 'teddy bear', 'baseball bat', 'skis'
]
test_cats = [
    'bicycle', 'boat', 'stop sign', 'bird', 'backpack', 'frisbee', 'snowboard',
    'surfboard', 'cup', 'fork', 'spoon', 'broccoli', 'chair', 'keyboard', 'microwave',
    'vase'
]
# assert not (train_cats & val_cats or train_cats & test_cats or val_cats & test_cats)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, phase, coco_jsons, data_dir, augment, repeats=1):
        super().__init__()
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        # Prepare image_paths and labels
        cats = {'train': train_cats, 'val': val_cats, 'test': test_cats}[phase]
        # idx2cat = sorted(list(cats))
        idx2cat = list(cats)
        cat2idx = dict([(cat, idx) for idx, cat in enumerate(idx2cat)])
        image_paths = []
        labels = []
        for coco_json in coco_jsons:
            with open(coco_json, 'r') as reader:
                data_info = eval(reader.read())
            for image_name in data_info.keys():
                if set(data_info[image_name]['categories']) - set(cats):
                    continue
                label = torch.zeros(1, len(cats), dtype=torch.float)
                for cat in data_info[image_name]['categories']:
                    label[0, cat2idx[cat]] = 1
                image_paths.append(data_dir + '/' + image_name.split('_')[1] + '/' + image_name)
                labels.append(label)
        self.image_paths = image_paths
        self.labels = torch.cat(labels, dim=0)  # self.labels.size() = (N, num_classes)
        # Build augmentations
        aug = []
        if augment:
            aug += [
                transforms.RandomAffine(0, translate=(0.03, 0.03)),
                transforms.ColorJitter(contrast=0.25),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        aug += [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.aug_func = transforms.Compose(aug)
        self.repeats = repeats
        self.info = (
            f'{phase} dataset:\n'
            f'From coco_jsons: {coco_jsons}\n'
            f'There are {len(self.image_paths)} images and {len(cats)} classes, with {repeats} repeats\n'
            f'Augmentations are {aug}\n'
            f'Label counts:\n'
            f'{idx2cat}\n'
            f'{torch.sum(self.labels, dim=0).numpy().tolist()}\n'
        )

    def __len__(self):
        return len(self.image_paths) * self.repeats

    def __getitem__(self, index):
        image_path = self.image_paths[index % len(self.image_paths)]
        image = Image.open(self.image_paths[index % len(self.image_paths)])
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.aug_func(image)
        label = self.labels[index % len(self.image_paths)]

        return image, label, index, image_path, False
