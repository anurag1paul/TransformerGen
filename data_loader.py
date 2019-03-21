import os
from abc import ABC, abstractmethod

import numpy
import numpy as np
import torch
import pandas as pd

from PIL import Image
from nltk import RegexpTokenizer
from torch.utils.data import Dataset
from torchvision import transforms
from pytorch_pretrained_bert.tokenization import BertTokenizer

from data_preprocess import CubDataPreprocessor, FlickrDataPreprocessor


def prepare_data(data, device):
    imgs, captions, class_ids, caption_lengths = data

    real_imgs = []
    for i in range(len(imgs)):
        real_imgs.append(imgs[i].to(device))

    max_len = 30
    # captions = captions[:, :max_len]
    captions = captions.squeeze()
    captions = captions.to(device)

    class_ids = class_ids.numpy()

    caption_lengths = caption_lengths.numpy()
    mask = caption_lengths[:,None] > np.arange(max_len)
    input_mask = np.zeros(mask.shape)
    input_mask[mask] = 1
    input_mask = torch.from_numpy(input_mask).squeeze().to(device)

    return [real_imgs, captions, class_ids, input_mask]


def get_imgs(img_path, imsize, opts, bbox=None,
             transform=None):
    normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = Image.open(img_path).convert('RGB')
    width, height = img.size

    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []

    for i in range(opts.TREE.BRANCH_NUM):
        # print(imsize[i])
        re_img = transforms.Resize((imsize[i], imsize[i]))(img)
        ret.append(normalize(re_img))

    return ret


class AbstractTokenizer(ABC):

    def __init__(self, max_caption_size):
        self.max_caption_size =max_caption_size

    def get_padded_tensor(self, caption):
        unpadded = self.tokenize(caption)
        length = len(unpadded)
        if length > self.max_caption_size:
            out = unpadded[:self.max_caption_size]
            length = self.max_caption_size
        else:
            out = [0] * self.max_caption_size
            out[:length] = unpadded

        return torch.LongTensor(out), length

    @abstractmethod
    def tokenize(self, caption):
        pass


class DefaultCaptionTokenizer(AbstractTokenizer):

    def __init__(self, word_to_idx, max_caption_size):
        super().__init__(max_caption_size)
        self.word_to_idx = word_to_idx
        self.max_caption_size = max_caption_size

    def tokenize(self, caption):
        cap = caption.replace(u"\ufffd\ufffd", u" ")
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(cap.lower())

        tokens_new = []
        for t in tokens:
            t = t.encode('ascii', 'ignore').decode('ascii')
            if len(t) > 0:
                if t in self.word_to_idx:
                    tokens_new.append(self.word_to_idx[t])

        return tokens_new


class BertCaptionTokenizer(AbstractTokenizer):

    def __init__(self, max_caption_size):
        super().__init__(max_caption_size)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        print("Using Bert tokenizer")

    def tokenize(self, caption):
        unpadded = self.tokenizer.tokenize(caption)
        out = self.tokenizer.convert_tokens_to_ids(unpadded)
        return out


def prepare_data(data, device):
    imgs, captions, class_ids, caption_lengths = data

    real_imgs = []
    for i in range(len(imgs)):
        real_imgs.append(imgs[i].to(device))

    max_len = 30
    # captions = captions[:, :max_len]
    captions = captions.squeeze()
    captions = captions.to(device)

    class_ids = class_ids.numpy()

    caption_lengths = caption_lengths.numpy()
    mask = caption_lengths[:,None] > np.arange(max_len)
    input_mask = np.zeros(mask.shape)
    input_mask[mask] = 1
    input_mask = torch.from_numpy(input_mask).squeeze().to(device)

    return [real_imgs, captions, class_ids, input_mask]


def get_imgs(img_path, imsize, opts, bbox=None,
             transform=None):
    normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = Image.open(img_path).convert('RGB')
    width, height = img.size

    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []

    for i in range(opts.TREE.BRANCH_NUM):
        # print(imsize[i])
        re_img = transforms.Resize((imsize[i], imsize[i]))(img)
        ret.append(normalize(re_img))

    return ret


class AbstractTokenizer(ABC):

    def __init__(self, max_caption_size):
        self.max_caption_size =max_caption_size

    def get_padded_tensor(self, caption):
        unpadded = self.tokenize(caption)
        length = len(unpadded)
        if length > self.max_caption_size:
            out = unpadded[:self.max_caption_size]
            length = self.max_caption_size
        else:
            out = [0] * self.max_caption_size
            out[:length] = unpadded

        return torch.LongTensor(out), length

    @abstractmethod
    def tokenize(self, caption):
        pass


class DefaultCaptionTokenizer(AbstractTokenizer):

    def __init__(self, word_to_idx, max_caption_size):
        super().__init__(max_caption_size)
        self.word_to_idx = word_to_idx
        self.max_caption_size = max_caption_size

    def tokenize(self, caption):
        cap = caption.replace(u"\ufffd\ufffd", u" ")
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(cap.lower())

        tokens_new = []
        for t in tokens:
            t = t.encode('ascii', 'ignore').decode('ascii')
            if len(t) > 0:
                if t in self.word_to_idx:
                    tokens_new.append(self.word_to_idx[t])

        return tokens_new


class BertCaptionTokenizer(AbstractTokenizer):

    def __init__(self, max_caption_size):
        super().__init__(max_caption_size)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        print("Using Bert tokenizer")

    def tokenize(self, caption):
        unpadded = self.tokenizer.tokenize(caption)
        out = self.tokenizer.convert_tokens_to_ids(unpadded)
        return out


class CubDataset(Dataset):

    def __init__(self, preprocessor: CubDataPreprocessor, opts, tokenizer, mode="train"):
        super(CubDataset, self).__init__()
        self.preprocessor = preprocessor
        self.mode = mode
        self.max_caption_size = 30
        self.opts = opts
        self.tokenizer = tokenizer

        if mode == "train":
            self.img_file_names = self.preprocessor.get_train_files()
        elif mode == "val":
            self.img_file_names = self.preprocessor.get_val_files()
        else:
            self.img_file_names = self.preprocessor.get_test_files()

        self.img_captions = []
        self.filename_bbox, self.filename_class = self.load_data()

        for name in self.img_file_names:
            name_parts = name.split('.')
            txt_name = '.'.join(name_parts[0:-1]) + '.txt'
            txt_path = os.path.join(self.preprocessor.captions_path, txt_name)
            with open(txt_path,  encoding='utf-8') as captions_file:
                captions = captions_file.read().splitlines()
                self.img_captions.append(captions)

        self.imsize = []
        base_size = opts.TREE.BASE_SIZE
        for i in range(opts.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

    def __len__(self):
        return len(self.img_file_names)

    def load_data(self):
        bbox_path = os.path.join(self.preprocessor.data_dir, 'bounding_boxes.txt')
        class_path = os.path.join(self.preprocessor.data_dir, 'image_class_labels.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        classes = pd.read_csv(class_path,delim_whitespace=True,
                                        header=None)

        df_filenames = pd.read_csv(self.preprocessor.images_path_file,
                                   delim_whitespace=True, header=None)

        filenames = df_filenames[1].tolist()
        filename_bbox = {}
        filename_classes = {}
        numImgs = len(filenames)

        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i]
            filename_bbox[key] = bbox
            filename_classes[key] = classes[1][i]
        return filename_bbox, filename_classes

    def __getitem__(self, idx):
        image_name = os.path.join(self.preprocessor.data_path, self.img_file_names[idx])
        image = get_imgs(image_name, self.imsize, self.opts, bbox=self.filename_bbox[self.img_file_names[idx]])

        # select a random sentence
        cap_idx = np.random.choice(np.arange(len(self.img_captions[idx])))
        caption, caption_length = self.tokenizer.get_padded_tensor(self.img_captions[idx][cap_idx])
        class_id = numpy.array(self.filename_class[self.img_file_names[idx]])
        caption_length = numpy.array(caption_length)

        return image, caption, class_id, caption_length


class FlickrDataset(Dataset):

    def __init__(self, preprocessor: FlickrDataPreprocessor, opts, tokenizer, mode="train"):
        super(FlickrDataset, self).__init__()
        self.preprocessor = preprocessor
        self.mode = mode
        self.max_caption_size = 30
        self.opts = opts
        self.tokenizer = tokenizer

        if mode == "train":
            self.img_file_names = self.preprocessor.get_train_files()
        elif mode == "val":
            self.img_file_names = self.preprocessor.get_val_files()
        else:
            self.img_file_names = self.preprocessor.get_test_files()

        self.img_captions_dict = preprocessor.get_img_caption_files()

        self.imsize = []
        base_size = opts.TREE.BASE_SIZE
        for i in range(opts.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

    def __len__(self):
        return len(self.img_file_names)

    def __getitem__(self, idx):
        image_name = self.img_file_names[idx]
        image_path = os.path.join(self.preprocessor.data_path, image_name)
        image = get_imgs(image_path, self.imsize, self.opts)

        # select a random sentence
        cap_idx = np.random.choice(np.arange(len(self.img_captions_dict[image_name])))
        caption, caption_length = self.tokenizer.get_padded_tensor(self.img_captions_dict[image_name][cap_idx])
        class_id = numpy.array(idx)
        caption_length = numpy.array(caption_length)

        return image, caption, class_id, caption_length
