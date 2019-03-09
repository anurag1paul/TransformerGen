import os
import pickle

import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split


class DataPreprocessor:

    def __init__(self, dataset_name, data_dir):
        self.data_dir = data_dir
        self.images_path_file = os.path.join(data_dir, "images.txt")
        self.data_path = os.path.join(data_dir, "images")
        self.captions_path = os.path.join(data_dir, "text_c10")
        self.processed_data = "data/"

        # we load the path names of all image files
        with open(self.images_path_file, "r") as imf:
            self.file_names = imf.read().splitlines()

        self.bag_of_words_path = self.processed_data + dataset_name + "_bag_of_words.pkl"
        self.train_test_split_path = self.processed_data + dataset_name + "_train_test.pkl"

        if not os.path.exists(self.processed_data):
            os.makedirs(self.processed_data)

        self.bag_of_words = None
        if os.path.exists(self.bag_of_words_path):
            with open(self.bag_of_words_path, "rb") as bow_file:
                self.bag_of_words = pickle.load(bow_file)
        else:
            self.bag_of_words = self.prepare_data()

        self.train_test = None

        if os.path.exists(self.train_test_split_path):
            with open(self.train_test_split_path, "rb") as tt_file:
                self.train_test = pickle.load(tt_file)
        else:
            self.train_test = self.train_test_split()

    def get_word_to_idx(self):
        return self.bag_of_words["word_to_idx"]

    def get_idx_to_word(self):
        return self.bag_of_words["idx_to_word"]

    def get_train_files(self):
        return self.train_test["train"]

    def get_test_files(self):
        return self.train_test["test"]

    def get_val_files(self):
        return self.train_test["val"]

    def prepare_data(self):

        all_captions =[]

        for img in self.file_names:
            name_parts = img.split()
            file_name = name_parts[1]

            txt_name = '.'.join(file_name.split('.')[0:-1]) + '.txt'
            txt_path = os.path.join(self.captions_path, txt_name)
            with open(txt_path, 'r', encoding='utf-8') as txt_file:
                captions = txt_file.read().splitlines()

                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace(u"\ufffd\ufffd", u" ")

                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())

                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.extend(tokens_new)

        vocab = np.unique(all_captions)

        idx_to_word = dict()
        idx_to_word[0] = '<end>'
        word_to_idx = dict()
        word_to_idx['<end>'] = 0
        idx = 1

        for w in vocab:
            word_to_idx[w] = idx
            idx_to_word[idx] = w
            idx += 1

        bag_of_words = {"idx_to_word": idx_to_word, "word_to_idx": word_to_idx}
        with open(self.bag_of_words_path, "wb") as f:
            pickle.dump(bag_of_words, f)

        return bag_of_words

    def train_test_split(self):

        class_path = os.path.join(self.data_dir, 'image_class_labels.txt')
        classes = pd.read_csv(class_path,delim_whitespace=True,
                                header=None)
        filepath = os.path.join(self.data_dir, 'images.txt')
        df_filenames = pd.read_csv(filepath,
                           delim_whitespace=True, header=None)

        filenames = df_filenames[1].tolist()
        labels = classes[1].tolist()

        Xtrain, Xtest, ytrain, ytest = train_test_split(filenames, labels, test_size=0.1)
        Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, test_size=0.075)

        file_names = {"train": Xtrain, "val":Xval, "test": Xtest}

        with open(self.train_test_split_path, "wb" ) as tt_file:
            pickle.dump(file_names, tt_file)

        return file_names


# if __name__ == "__main__":
#     preprocessor = DataPreprocessor(img_name_path, data_path, captions_path)
#     print(len(preprocessor.get_word_to_idx()))
