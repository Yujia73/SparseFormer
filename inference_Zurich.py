import argparse
import numpy as np
import torch
import os
import torch.backends.cudnn as cudnn
from tqdm import tqdm 
from skimage.io import imread, imsave
import torchvision.transforms.functional as TF
from osgeo import gdal
from osgeo.gdalconst import GDT_Byte, GDT_Float32, GDT_UInt16
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
from collections import OrderedDict
from network.SparseFormer_model import SparseFormer
from utils.PAR import PAR

import torch.nn as nn


def main():

    data_root = './data/Zurich/img/'
    ckpt = './Result_Zurich/time1006_1651/XXX.pth'
    txt = './data/Zurich/test.txt'
    file = open(txt, 'r')

    num_classes=8

    model_name = 'SparseFormer'
    save_root = './prediction/Zurich/'

    img_paths = list(file)

    if os.path.exists(save_root)==False:
        os.makedirs(save_root)
    
    cudnn.enabled = True
    cudnn.benchmark = True
           
    # ----- create model ----- # 
   
    model = SparseFormer(num_classes=num_classes,pretrained=False,attn1='CA',attn2='CBAM').cuda()     
        
    state_dict1 =torch.load(ckpt)
    model.load_state_dict(state_dict1,strict=False)
    model.eval()

    block_size =512,512
    min_overlap = 100

    for img_path in tqdm(img_paths):
        torch.cuda.empty_cache()
        img_path = img_path.replace('\n','')
        RGBimg=gdal.Open(str(data_root+img_path))
        band=RGBimg.GetRasterBand(1)                                                 
 
        image = imread(str(data_root+img_path))
        #image = np.asarray(image, np.float32)
        image = np.array(image,dtype='uint8') 
        image_size = image.shape[0:2]
        
        y_end,x_end = np.subtract(image_size, block_size)
        x = np.linspace(0, x_end, int(np.ceil(x_end/np.float64(block_size[1]-min_overlap)))+1, endpoint=True).astype('int')
        y = np.linspace(0, y_end, int(np.ceil(y_end/np.float64(block_size[0]-min_overlap)))+1, endpoint=True).astype('int')
        
        test_pred = np.zeros(image_size) 
        image=TF.to_tensor(image).cuda().unsqueeze(0)
        for j in range(len(x)):    
            for k in range(len(y)):            
                r_start,c_start = (y[k],x[j])
                r_end,c_end = (r_start+block_size[0],c_start+block_size[1])               
                image_part = image[0,:,r_start:r_end, c_start:c_end].unsqueeze(0).cuda()
                  
                with torch.no_grad():    
                    output = model(image_part)

                _,pred1 = torch.max(torch.softmax(output[2],1).detach(), 1)
                pred = pred1.squeeze().data.cpu().numpy()

                if (j==0)and(k==0):
                    test_pred[r_start:r_end, c_start:c_end] = pred
                elif (j==0)and(k!=0):
                    test_pred[r_start+int(min_overlap/2):r_end, c_start:c_end] = pred[int(min_overlap/2):,:]
                elif (j!=0)and(k==0):
                    test_pred[r_start:r_end, c_start+int(min_overlap/2):c_end] = pred[:,int(min_overlap/2):]
                elif (j!=0)and(k!=0):
                    test_pred[r_start+int(min_overlap/2):r_end, c_start+int(min_overlap/2):c_end] = pred[int(min_overlap/2):,int(min_overlap/2):]


        test_pred = np.asarray(test_pred, dtype=np.uint8)

        # 1：创建新图像容器
        rgb_image = np.zeros(shape=(*test_pred.shape, 3))
        # 2： 遍历每个像素点
        
        rgb_image[test_pred== 0, :] = [0,0,0]
        rgb_image[test_pred== 1, :] = [100,100,100]
        rgb_image[test_pred== 2, :] = [0,125,0]
        rgb_image[test_pred== 3, :] = [0,255,0]        
        rgb_image[test_pred== 4, :] = [150,80,0]
        rgb_image[test_pred== 5, :] = [0,0,150]
        rgb_image[test_pred== 6, :] = [255,255,0]
        rgb_image[test_pred== 7, :] = [150,150,255]
        rgb_image[test_pred== 8, :] = [255,255,255]


        rgb_image = np.transpose(rgb_image, (2, 0, 1))
        save_name = save_root+img_path
        gtiff_driver =gdal.GetDriverByName('GTiff')
        result = gtiff_driver.Create(save_name,int(band.XSize), int(band.YSize), 3, GDT_Byte,options=["TILED=YES","COMPRESS=LZW"])         
        result.SetProjection(RGBimg.GetProjection())   
        result.SetGeoTransform(RGBimg.GetGeoTransform())
        result.WriteArray(rgb_image)
        result.FlushCache()

        save_name1 = save_root+'gray_'+img_path
        gtiff_driver =gdal.GetDriverByName('GTiff')
        result = gtiff_driver.Create(save_name1,band.XSize, band.YSize, 1, GDT_Byte,options=["TILED=YES","COMPRESS=LZW"])         
        result.SetProjection(RGBimg.GetProjection())   
        result.SetGeoTransform(RGBimg.GetGeoTransform())
        result.WriteArray(test_pred)
        result.FlushCache()
        
if __name__ == '__main__':
    main()
