import os

import cv2
import numpy as np
import torch
import torch.utils.data

#DataLoader类的__iter__方法，该方法里面再调用DataLoaderIter类的初始化操作__init__。而当执行for循环操作时，调用DataLoaderIter类的__next__方法，在该方法中通过self.collate_fn接口读取self.dataset数据时就会调用TSNDataSet类的__getitem__方法
class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img_3d = np.zeros((4,512,512,3), np.uint8)
        img_gray_3d = np.zeros((4,512,512,1), np.uint8)
        for t in range(4):
            img_3d[t] = cv2.imread(os.path.join(self.img_dir, 'ori_' + img_id + '_' + str(t+1) + self.img_ext))#读原图
            img_gray_3d[t] = cv2.imread(os.path.join(self.img_dir, 'ori_' + img_id + '_' + str(t+1) + self.img_ext), cv2.IMREAD_GRAYSCALE)[..., None]
            img_gray_3d[t]= np.array(img_gray_3d[t])

        mask = []
        for i in range(self.num_classes):
            print(img_id)
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                        'gt_' + img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)#读hand segmentation
        # mask = cv2.imread(os.path.join(self.mask_dir, img_id + self.mask_ext))
        # mask = np.array(mask)

        if self.transform is not None:
            for au in range(4):
                augmented_3d = self.transform(image=img_3d[au], mask=mask, image_gray=img_gray_3d[au])
                img_3d[au] = augmented_3d['image']
                mask = augmented_3d['mask']
                img_gray_3d[au] = augmented_3d['image_gray']
        
        img_3d = img_3d.astype('float32') / 255
        img_3d = img_3d.transpose(3, 0, 1, 2)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        img_gray_3d = img_gray_3d.astype('float32') / 255
        img_gray_3d = img_gray_3d.transpose(3, 0, 1, 2)

        return img_3d, mask, img_gray_3d, {'img_id': img_id}
