import os
import torch.utils.data as data
from torchvision import transforms as T
from PIL import Image
import numpy as np
import cv2
import time


class DataSource(data.Dataset):
    def __init__(self, root, resize=256, crop_size=224, train=True):
        self.root = os.path.expanduser(root)
        self.resize = resize
        self.crop_size = crop_size
        self.train = train

        self.image_poses = []
        self.images_path = []

        self._get_data()

        # TODO: Define preprocessing

        # Load mean image
        self.mean_image_path = os.path.join(self.root, 'mean_image.npy')
        if os.path.exists(self.mean_image_path):
            self.mean_image = np.load(self.mean_image_path)
            print("Mean image loaded!")
        else:
            self.mean_image = self.generate_mean_image()

    def _get_data(self):

        if self.train:
            txt_file = self.root + 'dataset_train.txt'
        else:
            txt_file = self.root + 'dataset_test.txt'

        with open(txt_file, 'r') as f:
            next(f)  # skip the 3 header lines
            next(f)
            next(f)
            for line in f:
                fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
                p0 = float(p0)
                p1 = float(p1)
                p2 = float(p2)
                p3 = float(p3)
                p4 = float(p4)
                p5 = float(p5)
                p6 = float(p6)
                self.image_poses.append((p0, p1, p2, p3, p4, p5, p6))
                self.images_path.append(self.root + fname)

    def generate_mean_image(self):
        print("Computing mean image:")
        start = time.time()
        # TODO: Compute mean image

        # Initialize mean_image
        w, h = Image.open(self.images_path[0]).size
        mean_image = np.zeros((h, w, 3), dtype=np.float64)

        # Iterate over all training images
        # Resize, Compute mean, etc...
        N = len(self.images_path)
        for image_path in self.images_path:
            # Load each image
            img = Image.open(image_path)

            # Resize to required dimensions
            if h < w:
                h_r = self.resize / h
                img = img.resize((int(w * h_r), self.resize), Image.NEAREST)
            else:
                w_r = self.resize / w
                img = img.resize((self.resize, int(h * w_r)), Image.NEAREST)

            img = np.array(img, dtype=np.float64) / 255
            # Add to mean
            if mean_image.shape == img.shape:
                mean_image = (mean_image + img)
            else:
                mean_image = np.zeros(img.shape, dtype=np.float64)
                mean_image = (mean_image + img)

        mean_image = mean_image/N

        # Store mean image
        # mean_image = np.array(np.round(mean_image), dtype=np.uint8)
        np.save('mean_image', mean_image)
        img = Image.fromarray(
            np.array(mean_image*255, dtype=np.uint8), mode='RGB')
        img.save('mean_image.png')
        end = time.time()

        print(f"Mean image computed in {end-start} seconds!")

        return mean_image

    def __getitem__(self, index):
        """
        return the data of one image
        """
        img_path = self.images_path[index]
        img_pose = self.image_poses[index]

        data = Image.open(img_path)

        # TODO: Perform preprocessing
        # Resize
        resize_transform = T.Resize(size=self.resize)
        data = resize_transform(data)

        # Subtracting mean Image using numpy
        data_numpy = np.array(data, dtype=np.float64) / 255

        data_numpy = np.subtract(data_numpy, self.mean_image)

        data_numpy = (data_numpy - np.amin(data_numpy)) / \
            (np.amax(data_numpy) - np.amin(data_numpy))

        data = Image.fromarray(np.array(data_numpy * 255, dtype=np.uint8))

        # Crop
        if self.train:
            crop_transform = T.RandomCrop((224, 224))
        else:
            crop_transform = T.CenterCrop((224, 224))
        data = crop_transform(data)

        # To Tensor and Normalize
        normalize_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        data = normalize_transform(data)

        return data, img_pose

    def __len__(self):
        return len(self.images_path)
