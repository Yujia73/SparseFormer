import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as tf
import cv2
import random
import albumentations as A

IMG_MEAN = np.array((83.00327463, 91.67919424, 79.05708592), dtype=np.float32)

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]


def random_crop(image,mask,gt,crop_size=(512,512)):
    not_valid = True
    n = 0
    while not_valid:
        i, j, h, w = transforms.RandomCrop.get_params(image,output_size=crop_size)
        image_crop = tf.crop(image,i,j,h,w)
        mask_crop = tf.crop(mask,i,j,h,w)
        gt_crop = tf.crop(gt,i,j,h,w)
            
        label = np.asarray(mask_crop, np.float32)       
        n=n+1

        if np.sum(label!=255)>1:
            not_valid = False

    return image_crop,mask_crop,gt_crop

#training dataset
class ISPRSDataSet(data.Dataset):
    def __init__(self, root, list_path, crop_size=(512, 512), mean=IMG_MEAN, scale=False, mirror=False, ignore_label=255,set='P',id=11,mode=0):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.set = set
        self.mode = mode
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.type = set
        self.id = id
        
        n_repeat = 50
        self.img_ids = self.img_ids * n_repeat

        self.files = []


        for name in self.img_ids:
            img_file = osp.join(self.root, "img/%s" % name)
            label_file = osp.join(self.root, str(self.type)+"/an"+str(self.id)+"/mask_"+name)       
            gt_file = osp.join(self.root, "gt/"+name) #ground truth
            self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "gt": gt_file,
                    "name": name
                })
           

    def _rotation(self,img,gt,label):
        
        index = random.randint(1, 6)
        if index == 1:
            new_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            new_gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
            new_label = cv2.rotate(label, cv2.ROTATE_90_CLOCKWISE)
            
        elif index ==2:
            new_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            new_gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
            new_label = cv2.rotate(label, cv2.ROTATE_90_COUNTERCLOCKWISE)

        elif index == 3:
            new_img = cv2.rotate(img, cv2.ROTATE_180)
            new_gt = cv2.rotate(gt, cv2.ROTATE_180)
            new_label = cv2.rotate(label, cv2.ROTATE_180)

        elif index == 4:
            new_img = cv2.flip(img,1)
            new_gt = cv2.flip(gt,1)
            new_label = cv2.flip(label,1)

        elif index==5:
            new_img = cv2.flip(img,0)
            new_gt = cv2.flip(gt,0)
            new_label = cv2.flip(label,0)   
        else:
            new_img = img
            new_gt = gt
            new_label = label

        return new_img,new_gt,new_label

    def make_clslabel(self, label,num_classes=5,ingore_index=255):
        label_set = np.unique(label)
        label_set = label_set.astype(int)
        cls_label = np.zeros(num_classes)
        cls_label = cls_label.astype(int)
        for i in label_set:
            if i < ingore_index:
                cls_label[i] += 1
        return cls_label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"]).convert('P')
        gt = Image.open(datafiles["gt"]).convert('P')
            
        image1,label1,gt1 = random_crop(image,label,gt,self.crop_size)

        if (image,label) == (0,0):
            return (0,0)

        label1 = np.asarray(label1, np.float32)
        image1 = np.asarray(image1)
        gt1 = np.asarray(gt1,np.float32)

        label1[label1>4]=255
        gt1[gt1>4]=255
                
        image_aug,gt_aug,label_aug  = self._rotation(image1,gt1,label1)             

        image_aug = tf.to_tensor(image_aug)
        image_aug = image_aug.numpy()
        
        cls_label = self.make_clslabel(label1,num_classes=5,ingore_index=255)
        
        return image_aug.copy(),label_aug.copy(),gt_aug.copy()#,cls_label.copy()
    
    


#test  dataset
class ISPRSTestDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None,crop_size=(512, 512), mean=IMG_MEAN, scale=False, mirror=False, ignore_label=255,set='train',mode=0):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.set = set
        self.mode = mode
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        
        if not max_iters==None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]

        self.files = []


        for name in self.img_ids:
            img_file = osp.join(self.root, "img/%s" % name)
            gt_file = osp.join(self.root, "gt/"+name)
            self.files.append({
                    "img": img_file,
                    "gt": gt_file,
                    "name": name
                })
           
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        gt = Image.open(datafiles["gt"]).convert('P')
            
        name = datafiles["name"]
        image = np.asarray(image)
        gt = np.asarray(gt,np.float32)

        gt[gt>4]=255
        
        image = tf.to_tensor(image)
        image = image.numpy()
  
        return image.copy(), gt.copy(),name
