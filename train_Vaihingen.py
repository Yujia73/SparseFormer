import argparse
import os
import random
import time
from torch.utils import data
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm
from torch.utils import data
from dataloaders.ISPRSDataset import ISPRSDataSet,ISPRSTestDataSet
from network.SparseFormer_model import SparseFormer
from tqdm import tqdm
import os
from utils.mytool import *
from utils.PAR import PAR
import warnings
warnings.filterwarnings("ignore")

os.environ['KMP_DUPLICATE_LIB_OK']='True'


name_classes = np.array(['impervious surface','Building','Low vegetation','Tree','Car'], dtype=np.str_) #Vaihingen categories
epsilon = 1e-14

def init_seeds(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 


parser = argparse.ArgumentParser()

parser.add_argument("--data_name", type=str, default='point_an1_Vaihingen_p0.5_r0.3_only_ce',
                        help="dataset path.")
parser.add_argument("--model_name", type=str, default='SparseFormer',
                        help="dataset path.")

parser.add_argument("--data_dir", type=str, default='./data/Vaihingen/',
                        help="dataset path.")
parser.add_argument("--test_dir", type=str, default='./data/Vaihingen/',
                        help="dataset path.")
parser.add_argument("--train_list", type=str, default='./data/Vaihingen/train.txt',
                        help="training list file.")
parser.add_argument("--test_list", type=str, default='./data/Vaihingen/test.txt',
                        help="test list file.")  
parser.add_argument('--sup_type', type=str,
                    default='point', help='supervision type') #label type(line,point,polygon)
parser.add_argument('--sup_id', type=str,
                    default='11', help='supervision type') #label type(11,22,33,44)

parser.add_argument('--num_classes', type=int,  default=5,
                    help='output channel of network')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')


parser.add_argument('--attn1', type=str,  default='CA',
                    help='output channel of network')
parser.add_argument('--attn2', type=str,  default='CBAM',
                    help='output channel of network')

parser.add_argument('--cfg', type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')
parser.add_argument("--snapshot_dir", type=str, default='./Result_Vaihingen/',
                        help="where to save snapshots of the model.")

args = parser.parse_args()

modename = ['full','point']

def main():

    snapshot_dir = args.snapshot_dir+'/time'+time.strftime('%m%d_%H%M', time.localtime(time.time()))+'/'
    if os.path.exists(snapshot_dir)==False:
        os.makedirs(snapshot_dir)
    f = open(snapshot_dir+'train_log.txt', 'w')

    init_seeds()
    
    # Create network   
    # ----- create model ----- #
    model = SparseFormer(num_classes=args.num_classes, drop_rate=0.4, normal_init=True, pretrained=True,attn1=args.attn1,attn2=args.attn2).cuda()
    params = model.parameters()
    
    optimizer = torch.optim.Adam(params, args.base_lr, betas=(args.beta1, args.beta2))
    
    trainloader = data.DataLoader(
                        ISPRSDataSet(args.data_dir, args.train_list,
                        crop_size=(512,512),set=args.sup_type,mode=0,id= args.sup_id),
                        batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    test_loader = data.DataLoader(
                        ISPRSTestDataSet(args.test_dir, args.test_list, set='test'),
                        batch_size=1, shuffle=False, num_workers=0, pin_memory=True) 

    optimizer.zero_grad()

        
    loss_hist = np.zeros((60000,10))
    F1_best = 0.4

    ce_loss = CrossEntropyLoss(ignore_index=255)
    kd_loss = KDLoss(T=10)

    model.train()

    par = PAR(dilations=[1,4,8,12,24],num_iter=10).cuda()

    for epoch in range(50):

        for batch_index,src_data in enumerate(tqdm(trainloader)):
            
            torch.cuda.empty_cache()
            model.train()

            tem_time = time.time()

            image,label_batch,gt_batch = src_data 
            image,label_batch, gt_batch= image.cuda(),label_batch.cuda().long(),gt_batch.long()

            # start training
            outputs1,outputs2,outputs3 = model(image)

            #warm-up stage
            if (epoch + 1)<=10:
                #supervised loss using point label
                loss_ce1 = ce_loss(outputs1,label_batch)
                loss_ce2 = ce_loss(outputs2,label_batch)
                loss_ce3 = ce_loss(outputs3,label_batch)                 
                
                ave_output = (outputs1+outputs2)/2
                loss_kl1 = kd_loss(outputs1,ave_output.detach())
                loss_kl2 = kd_loss(outputs2,ave_output.detach())
                
                loss_seg = (loss_ce1 + loss_ce2 + loss_ce3)/3
                loss_kl = (loss_kl1 + loss_kl2)/2

            else:                
                #credible pseudo-label learning    
                outputs1_par = par(image,outputs1)
                outputs2_par = par(image,outputs2)
                
                merged_labels = select_confident_region(outputs1_par.detach(),outputs2_par.detach(),label_batch,thed=0.5)
                #merged_labels = merge_cnn_without_confidence(outputs1_par.detach(),outputs2_par.detach())
                                                                   
                #cross-decoder learning
                loss_kl1 = joint_optimization(outputs1, outputs2.detach(), outputs3.detach(),10)
                loss_kl2 = joint_optimization(outputs2, outputs1.detach(), outputs3.detach(),10)
                                             
                loss_ce1 = ce_loss(outputs1,label_batch)
                loss_ce2 = ce_loss(outputs2,label_batch)
                loss_ce3 = structure_loss(outputs3,merged_labels,args.num_classes)

                loss_seg = (loss_ce1 + loss_ce2 + loss_ce3)/3   
                loss_kl = (loss_kl1 + loss_kl2)/2
                

            loss = loss_seg + 0.3*loss_kl #total loss
            
            optimizer.zero_grad()  
            loss.backward()
            optimizer.step()

            predict_labels1 = torch.argmax(outputs1,1)
            predict_labels2 = torch.argmax(outputs2,1)
            predict_labels3 = torch.argmax(outputs3,1)
            
            lbl_pred1 = predict_labels1.detach().cpu().numpy()
            lbl_pred2 = predict_labels2.detach().cpu().numpy()        
            lbl_pred3 = predict_labels3.detach().cpu().numpy()
                    
            lbl_gt = gt_batch.detach().cpu().numpy()\
            
            metrics_batch1 = []
            metrics_batch2 = []        
            metrics_batch3 = []              
                
            #training accuracy
            for lt, lp in zip(lbl_gt, lbl_pred1):
                _,_,mean_iu,_ = label_accuracy_score(lt, lp, n_class=args.num_classes)
                metrics_batch1.append(mean_iu)                
            miou1 = np.nanmean(metrics_batch1, axis=0)
            
            for lt, lp in zip(lbl_gt, lbl_pred2):
                _,_,mean_iu,_ = label_accuracy_score(lt, lp, n_class=args.num_classes)
                metrics_batch2.append(mean_iu)               
            miou2 = np.nanmean(metrics_batch2, axis=0)   
                
            for lt, lp in zip(lbl_gt, lbl_pred3):
                _,_,mean_iu,_ = label_accuracy_score(lt, lp, n_class=args.num_classes)
                metrics_batch3.append(mean_iu)              
            miou3 = np.nanmean(metrics_batch3, axis=0)            
            
            loss_hist[batch_index,0] = loss.item()
            loss_hist[batch_index,1] = loss_seg.item()  
            loss_hist[batch_index,2] = loss_ce3.item()  
            loss_hist[batch_index,3] = loss_kl.item()          
            loss_hist[batch_index,4] = miou1
            loss_hist[batch_index,5] = miou2
            loss_hist[batch_index,6] = miou3
            
            loss_hist[batch_index,-1] = time.time() - tem_time

            if (batch_index+1) % 20 == 0: 
                print('[Epoch %d | Iter %d] time: %.2f miou_p1 = %.1f miou_p2= %.1f miou_p3= %.1f loss = %.3f loss_seg = %.3f loss_ce3 = %.3f loss_kl = %.3f\n'%(epoch,batch_index+1,np.mean(loss_hist[batch_index-9:batch_index+1,-1]),
                                                                                                                    np.mean(loss_hist[batch_index-9:batch_index+1,4])*100,
                                                                                                                    np.mean(loss_hist[batch_index-9:batch_index+1,5])*100,
                                                                                                                    np.mean(loss_hist[batch_index-9:batch_index+1,6])*100,

                                                                                                                    np.mean(loss_hist[batch_index-9:batch_index+1,0]),
                                                                                                                    np.mean(loss_hist[batch_index-9:batch_index+1,1]),
                                                                                                                    np.mean(loss_hist[batch_index-9:batch_index+1,2]),
                                                                                                                    np.mean(loss_hist[batch_index-9:batch_index+1,3]),))
                        
                f.write('[Epoch %d | Iter %d] time: %.2f miou_p1 = %.1f miou_p2= %.1f miou_p3= %.1f loss = %.3f loss_seg = %.3f loss_ce3 = %.3f loss_kl = %.3f\n'%(epoch,batch_index+1,np.mean(loss_hist[batch_index-9:batch_index+1,-1]),
                                                                                                            np.mean(loss_hist[batch_index-9:batch_index+1,4])*100,
                                                                                                            np.mean(loss_hist[batch_index-9:batch_index+1,5])*100,
                                                                                                            np.mean(loss_hist[batch_index-9:batch_index+1,6])*100,
                                                                                                            np.mean(loss_hist[batch_index-9:batch_index+1,0]),
                                                                                                            np.mean(loss_hist[batch_index-9:batch_index+1,1]),
                                                                                                            np.mean(loss_hist[batch_index-9:batch_index+1,2]),
                                                                                                            np.mean(loss_hist[batch_index-9:batch_index+1,3]),))
                f.flush()
                    
        # evaluation per 2 epoch
        if (epoch+1) % 2  == 0:            
            model.eval()

            TP_all = np.zeros((args.num_classes, 1))
            FP_all = np.zeros((args.num_classes, 1))
            TN_all = np.zeros((args.num_classes, 1))
            FN_all = np.zeros((args.num_classes, 1))
            n_valid_sample_all = 0
            F1 = np.zeros((args.num_classes, 1))
            IoU = np.zeros((args.num_classes, 1))
        
            for index, batch in enumerate(test_loader):  
                image,label,name = batch
                label = label.squeeze().numpy()

                img_size = image.shape[2:] 
                block_size = 512,512
                min_overlap = 100

                # crop the test images into 512×512 patches
                y_end,x_end = np.subtract(img_size, block_size)
                x = np.linspace(0, x_end, int(np.ceil(x_end/np.float_(block_size[1]-min_overlap)))+1, endpoint=True).astype('int')
                y = np.linspace(0, y_end, int(np.ceil(y_end/np.float_(block_size[0]-min_overlap)))+1, endpoint=True).astype('int')

                test_pred1 = np.zeros(img_size)

                for j in range(len(x)):    
                    for k in range(len(y)):            
                        r_start,c_start = (y[k],x[j])
                        r_end,c_end = (r_start+block_size[0],c_start+block_size[1])
                        image_part = image[0,:,r_start:r_end, c_start:c_end].unsqueeze(0).cuda()

                        with torch.no_grad():
                            outputs1,outputs2,outputs3 = model(image_part)
                        
                        _,pred1 = torch.max(outputs3, 1)

                        pred1 = pred1.squeeze().data.cpu().numpy()


                        if (j==0)and(k==0):
                            test_pred1[r_start:r_end, c_start:c_end] = pred1

                        elif (j==0)and(k!=0):
                            test_pred1[r_start+int(min_overlap/2):r_end, c_start:c_end] = pred1[int(min_overlap/2):,:]

                        elif (j!=0)and(k==0):
                            test_pred1[r_start:r_end, c_start+int(min_overlap/2):c_end] = pred1[:,int(min_overlap/2):]

                        elif (j!=0)and(k!=0):
                            test_pred1[r_start+int(min_overlap/2):r_end, c_start+int(min_overlap/2):c_end] = pred1[int(min_overlap/2):,int(min_overlap/2):]

                
                print(index+1, '/', len(test_loader), ': Testing ', name)
                
                # evaluate one image
                TP,FP,TN,FN,n_valid_sample = eval_image(test_pred1.reshape(-1),label.reshape(-1),args.num_classes)
                TP_all += TP
                FP_all += FP
                TN_all += TN
                FN_all += FN
                n_valid_sample_all += n_valid_sample

            OA = np.sum(TP_all)*1.0 / n_valid_sample_all 
            for i in range(args.num_classes):
                P = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + epsilon)
                R = TP_all[i]*1.0 / (TP_all[i] + FN_all[i] + epsilon)
                F1[i] = 2.0*P*R / (P + R + epsilon)
                IoU[i] = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + FN_all[i] + epsilon)
            
            for i in range(args.num_classes):
                f.write('===>' + name_classes[i] + ': %.2f\n'%(F1[i] * 100))
                print('===>' + name_classes[i] + ': %.2f'%(F1[i] * 100))
            mF1 = np.mean(F1)
            mIoU = np.mean(IoU)
                        
            f.write('===> mean F1: %.2f mean IoU: %.2f OA: %.2f\n'%(mF1*100,mIoU*100,OA*100))
            print('===> mean F1: %.2f mean IoU: %.2f OA: %.2f'%(mF1*100,mIoU*100,OA*100))                    
            if mF1 > F1_best:     
                f.write('Save Model\n') 
                print('Save Model')
                F1_best = mF1                   
                model_name = args.model_name+'_'+args.data_name+'_Best_batch'+repr(epoch+1)+'mF1_'+repr(int(mF1*10000))+'.pth'
                torch.save(model.state_dict(), os.path.join(snapshot_dir, model_name))
  
    f.close()


if __name__ == '__main__':
    main()

