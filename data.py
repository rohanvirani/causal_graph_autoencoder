import numpy as np
import os
from os.path import join
from torch.utils import data


class CELEBA_LABELS(data.Dataset):
    """
    Args:
        root (string): Root directory of dataset where directory
            ``celebA`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, root, train=True, label=['Smiling','High_Cheekbones','Narrow_Eyes','Oval_Face']):
        attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
                      'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
                      'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                      'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
                      'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
                      'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
                      'Wearing_Necktie', 'Young']
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.filename = 'data'
        self.idx = []
        for i in np.arange(len(label)):
            self.idx.append(attributes.index(label[i]))
        print(self.idx)

        # now load the picked numpy arrays
        if self.train:
            train_labels = np.load(join(self.root, self.filename, 'yAllTrain.npy'))[100:, self.idx]
            self.train_labels = (train_labels.astype(int) + 1) // 2
            print(np.shape(self.train_labels))
            print(np.unique(self.train_labels))

        else:
            test_labels = np.load(join(self.root, self.filename, 'yAllTrain.npy'))[:100, self.idx]
            self.test_labels = (test_labels.astype(int) + 1) // 2

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        if self.train:
            target = self.train_labels[index]
        else:
            target = self.test_labels[index]

        return target

    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)
