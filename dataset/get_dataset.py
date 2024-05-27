import random
import numpy as np
from randaugment import RandAugment
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageDraw, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

np.set_printoptions(suppress=True)

class Mixed_handler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x = Image.open(self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)
    

class Mixed_mask_handler(Dataset):
    def __init__(self, X, Y, mask, transform=None):
        self.X = X
        self.Y = Y
        self.Mask = mask
        self.transform = transform

    def __getitem__(self, index):
        x = Image.open(self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        mask = self.Mask[index]
        return x, y, mask

    def __len__(self):
        return len(self.X)

def get_datasets(args, logger):

    train_transform = TransformUnlabeled_WS(args)
    test_transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()])

    path = args.dataset_dir+'/'+args.dataset_name+'/'+str(args.seed)
    
    lb_train_imgs = np.load(f'{path}/lb_image_{args.lb_ratio}.npy')
    lb_train_labels = np.load(f'{path}/lb_label_{args.lb_ratio}.npy')
    ub_train_imgs = np.load(f'{path}/ub_image_{args.lb_ratio}.npy')
    ub_train_labels = np.load(f'{path}/ub_label_{args.lb_ratio}.npy')
    test_imgs = np.load(f'{path}/test_image.npy')
    test_labels = np.load(f'{path}/test_label.npy')
    logger.info('load data success!')

    lb_train_dataset = Mixed_handler(
        lb_train_imgs,
        lb_train_labels,
        transform = train_transform
    )
    ub_train_dataset = Mixed_handler(
        ub_train_imgs,
        ub_train_labels,
        transform = train_transform
    )
    test_dataset = Mixed_handler(
        test_imgs,
        test_labels,
        transform = test_transform
    )
    
    args.n_classes = lb_train_labels.shape[1]
    logger.info(f'n_classes {args.n_classes}')
    logger.info(f"len(lb_train_dataset) {len(lb_train_dataset)}") 
    logger.info(f"len(ub_train_dataset) {len(ub_train_dataset)}")
    logger.info(f"len(test_dataset) {len(test_dataset)}")
    return lb_train_dataset, ub_train_dataset, test_dataset


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


class TransformUnlabeled_WS(object):
    def __init__(self, args):

        self.weak = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
            ]
        )

        strong = [
            transforms.RandomHorizontalFlip(),
            transforms.Resize((args.img_size, args.img_size)),
            RandAugment(),
            transforms.ToTensor(),
        ]

        if args.cutout > 0:
            strong.insert(2, CutoutPIL(cutout_factor=args.cutout))
        self.strong = transforms.Compose(strong)

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return weak, strong
