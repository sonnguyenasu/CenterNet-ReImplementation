from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torch


class Base(Dataset):
    def __init__(self, image_path, label_path, num_class):
        self.image_path = image_path
        self.label_path = label_path
        self.images = [os.path.join(image_path, img)
                       for img in os.listdir(image_path)]
        self.num_class = num_class
    def gaussian_radius(self,det_size, min_overlap=0.7):
        height, width = det_size

        a1  = 1
        b1  = (height + width)
        c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1  = (b1 + sq1) / 2

        a2  = 4
        b2  = 2 * (height + width)
        c2  = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2  = (b2 + sq2) / 2

        a3  = 4 * min_overlap
        b3  = -2 * min_overlap * (height + width)
        c3  = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3  = (b3 + sq3) / 2
        return min(r1, r2, r3)
    def _convert_box_to_center_mask(self, boxes, height, width):
        return None

    def _convert_box_to_center(self, boxes):
        return None

    def __len__(self):
        return len(self.images)

    def __getitem(self, idx):
        pass


class CustomDataset(Base):
    def __init__(self, image_path, label_path, num_class):
        super(CustomDataset, self).__init__(image_path, label_path, num_class)
        self.labels = [os.path.join(label_path, img[:-3]+'txt')
                       for img in os.listdir(self.image_path)]

    def _convert_box_to_center(self, boxes):
        center = []
        for box in boxes:
            center_x = box[1]
            center_y = box[2]
            center.append([box[0], center_x//4, center_y//4])
        return center

    def _convert_box_to_center_mask(self, boxes, height, width):
        center_mask = np.zeros((height//4, width//4, self.num_class))
        intended_value = np.zeros((height//4, width//4))

        x = np.arange(width//4)
        y = np.arange(height//4)
        index_x, index_y = np.meshgrid(x, y)
        centers = self._convert_box_to_center(boxes)
        for center in centers:
            std = self.gaussian_radius((height//4,width//4))/3#np.sqrt(center[1]**2+center[2]**2)/6
            intended_value[:, :] = np.exp(-((index_x - center[1])**2 +
                                            (index_y-center[2])**2)/(2*std**2))
            center_mask[:, :, center[0]] = np.where(
                center_mask[:, :, center[0]] < intended_value,
                intended_value, center_mask[:, :, center[0]])
        return center_mask

    def _get_offset_mask(self, boxes, height, width):
        offset_mask = np.zeros((height//4, width//4, 2))
        for box in boxes:
          try:
            offset_mask[box[2]//4, box[1]//4, 0] = box[1]-4*box[1]//4
            offset_mask[box[2]//4, box[1]//4, 1] = box[2]-4*box[2]//4
          except:
            print(boxes)
        return offset_mask

    def _get_size_mask(self, boxes, height, width):
        size_mask = np.zeros((height//4, width//4, 2))
        for box in boxes:
            size_mask[box[2]//4, box[1]//4, 0] = box[3]  # width offset
            size_mask[box[2]//4, box[1]//4, 1] = box[4]  # height offset
        return size_mask

    def _resize(self, img, boxes):
        height, width = img.shape[:2]
        new_height = 416#height - (height % 4)
        new_width = 416#width - (width % 4)
        res = []
        for box in boxes:
          res.append([int(box[0]), int(box[1]*new_width),
                  int(box[2] * new_height), int(box[3]*new_width),
                  int(box[4]*new_height)])
        img = cv2.resize(img, (new_width, new_height))
        return img, res

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])

        img_labels = open(self.labels[idx], 'r').readlines()
        boxes = []
        # classes = []
        for label in img_labels:
            dat = [float(num) for num in label.split(' ')]
            boxes.append(dat)
        img, boxes = self._resize(img, boxes)
        
        height, width = img.shape[:2]
        center_mask = self._convert_box_to_center_mask(boxes, height, width)
        offset_mask = self._get_offset_mask(boxes, height, width)
        size_mask = self._get_size_mask(boxes, height, width)
        centers = self._convert_box_to_center(boxes)
        img = img/255.0
        return torch.Tensor(img).permute(2, 0, 1), \
            torch.Tensor(center_mask).permute(2, 0, 1),\
            torch.Tensor(offset_mask).permute(2, 0, 1), \
            torch.Tensor(size_mask).permute(2, 0, 1), \
            #torch.Tensor(centers)  # torch.Tensor(centers)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from util import draw, get_center_from_center_mask
    ds = CustomDataset('../train/images',
                       '../train/labels', num_class=6)
    sample_loader = DataLoader(ds, batch_size=4, shuffle=True)
    img_, center_mask_, offset_mask_, size_mask_, centers_ = next(
            iter(sample_loader))
    for i in range(4):
        
        img = (255*img_[i:i+1,:,:,:].permute(0, 2, 3, 1)).long().squeeze(0)
        center_mask = center_mask_[i:i+1,:,:,:].permute(0, 2, 3, 1).numpy()
        size_mask = size_mask_[i:i+1,:,:,:].permute(0, 2, 3, 1).numpy()
        offset_mask = offset_mask_[i:i+1,:,:,:].permute(0, 2, 3, 1).numpy()
        #centers = get_center_from_center_mask(center_mask)
        #centers = ce
        centers=  centers_[i]
        print(centers_[0][:,0])
        print(img.shape)
        print(center_mask.shape)
        print(offset_mask.shape)
        print(size_mask.shape)
        imag = draw(img[:, :, :], centers,
                    offset_mask[0, :, :, :], size_mask[0, :, :, :])
        fig, ax = plt.subplots(3, 3)
        ax[0, 0].imshow(center_mask[0,:, :, 0], cmap='gray')
        ax[0, 1].imshow(center_mask[0,:, :, 1], cmap='gray')
        ax[0, 2].imshow(center_mask[0,:, :, 2], cmap='gray')
        ax[1, 0].imshow(center_mask[0,:, :, 3], cmap='gray')
        ax[1, 1].imshow(center_mask[0,:, :, 4], cmap='gray')
        ax[1, 2].imshow(center_mask[0,:, :, 5], cmap='gray')
        ax[2, 0].imshow(imag[:, :, ::-1])  # [0, :, :, :])
        #ax[2, 1].imshow(img[0, :, :, :])
        ax[2, 1].imshow(size_mask[0,:, :, 0], cmap='gray')
        ax[2, 2].imshow(size_mask[0,:, :, 1], cmap='gray')

        #img = img.astype('uint8')

        #plt.imshow(img[:, :, 0], cmap='gray')
        plt.savefig('log.jpg')
