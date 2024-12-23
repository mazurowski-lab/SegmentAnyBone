#from segment_anything import SamPredictor, sam_model_registry
from models.sam import SamPredictor, sam_model_registry
from models.sam.utils.transforms import ResizeLongestSide
from models.sam.modeling.prompt_encoder import auto_cls_emb
from models.sam.modeling.prompt_encoder import attention_fusion
from skimage.measure import label
#Scientific computing 
import numpy as np
import os
#Pytorch packages
import torch
from torch import nn
import torch.optim as optim
from einops import rearrange
import torchvision
from torchvision import datasets
from tensorboardX import SummaryWriter
#Visulization
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
#Others
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import copy
from dataset_bone import MRI_dataset_multicls
import torch.nn.functional as F
from torch.nn.functional import one_hot
from pathlib import Path
from tqdm import tqdm
from losses import DiceLoss
from dsc import dice_coeff,dice_coeff_multi_class
import cv2
import monai
from utils import vis_image
import random

import cfg
args = cfg.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
args.if_mask_decoder_adapter=True
args.if_encoder_adapter = True
args.lr = 5e-4
args.decoder_adapt_depth = 2
args.if_warmup = True
args.initial_path = '/mnt/largeDrives/sevenTBTwo/bone_proj/codes_for_data/'
args.pretrain_weight = os.path.join('/mnt/largeDrives/sevenTBTwo/bone_proj/codes_for_data/588/fine-tune-sam/Medical-SAM-Adapter','2D-MobileSAM-onlyfusion-adapter_Bone_0107_paired_attentionpredicted','checkpoint_best.pth')
args.num_classes = 2
args.targets = 'multi_all'


def train_model(trainloader,valloader,dir_checkpoint,epochs):
    # Set up model
    
    if args.if_warmup:
        b_lr = args.lr / args.warmup_period
    else:
        b_lr = args.lr
    
    
    iter_num = 0
    max_iterations = epochs * len(trainloader) 
    writer = SummaryWriter(dir_checkpoint + '/log')
    
    sam = sam_model_registry["vit_t"](args,checkpoint=args.pretrain_weight,num_classes=args.num_classes) 
    sam.load_state_dict(torch.load(os.path.join(args.pretrain_weight)), strict = False)
    print(sam)
    
    for n, value in sam.named_parameters():
        value.requires_grad = False
    
    for n, value in sam.mask_decoder.named_parameters():
        if "Adapter" in n: # only update parameters in decoder adapter
            value.requires_grad = True
        if 'output_hypernetworks_mlps' in n:
            value.requires_grad = True
            
    print('if image encoder adapter:',args.if_encoder_adapter)
    print('if mask decoder adapter:',args.if_mask_decoder_adapter)
    sam.to('cuda')
    
    optimizer = optim.AdamW(sam.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay
    criterion1 = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, to_onehot_y=True,reduction='mean')
    criterion2 = nn.CrossEntropyLoss()
    
    pbar = tqdm(range(epochs))
    val_largest_dsc = 0
    last_update_epoch = 0
    for epoch in pbar:
        sam.train()
        train_loss = 0
        for i,data in enumerate(trainloader):
            imgs = data['image'].cuda()
            img_emb= sam.image_encoder(imgs)
            alpha = random.random()
            # automatic masks contaning all muscles
            msks = torchvision.transforms.Resize((args.out_size,args.out_size))(data['mask'])
            #print('mask unique value:',msks.unique())
            msks = msks.cuda()
            sparse_emb, dense_emb = sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
            pred, _ = sam.mask_decoder(
                            image_embeddings=img_emb,
                            image_pe=sam.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=sparse_emb,
                            dense_prompt_embeddings=dense_emb, 
                            multimask_output=True,
                          )
            loss_dice = criterion1(pred,msks.float()) 
            loss_ce = criterion2(pred,torch.squeeze(msks.long(),1))
            loss =  loss_dice + loss_ce
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if args.if_warmup and iter_num < args.warmup_period:
                lr_ = args.lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            else:
                if args.if_warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                    lr_ = args.lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
                        
            train_loss += loss.item()
            
            iter_num+=1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

        train_loss /= (i+1)
        pbar.set_description('Epoch num {}| train loss {} \n'.format(epoch,train_loss))

        if epoch%2==0:
            eval_loss=0
            dsc = 0
            sam.eval()
            with torch.no_grad():
                for i,data in enumerate(valloader):
                    imgs = data['image'].cuda()
                    img_emb= sam.image_encoder(imgs)
                    alpha = random.random()
                    msks = torchvision.transforms.Resize((args.out_size,args.out_size))(data['mask'])
                    msks = msks.cuda()
                    sparse_emb, dense_emb = sam.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                    )
                    pred, _ = sam.mask_decoder(
                                    image_embeddings=img_emb,
                                    image_pe=sam.prompt_encoder.get_dense_pe(), 
                                    sparse_prompt_embeddings=sparse_emb,
                                    dense_prompt_embeddings=dense_emb, 
                                    multimask_output=True,
                                  )
                    loss = criterion1(pred,msks.float()) + criterion2(pred,torch.squeeze(msks.long(),1))
                    eval_loss +=loss.item()
                    dsc_batch = dice_coeff_multi_class(pred.argmax(dim=1).cpu(), torch.squeeze(msks.long(),1).cpu().long(), 5)
                    dsc+=dsc_batch

                    
                eval_loss /= (i+1)
                dsc /= (i+1)
                
                writer.add_scalar('eval/loss', eval_loss, epoch)
                writer.add_scalar('eval/dice', dsc, epoch)
                
                print('Eval Epoch num {} | val loss {} | dsc {} \n'.format(epoch,eval_loss,dsc))
                if dsc>val_largest_dsc:
                    val_largest_dsc = dsc
                    last_update_epoch = epoch
                    print('largest DSC now: {}'.format(dsc))
                    Path(dir_checkpoint).mkdir(parents=True,exist_ok = True)
                    torch.save(sam.state_dict(),dir_checkpoint + '/checkpoint_best.pth')
                elif (epoch-last_update_epoch)>20:
                    # the network haven't been updated for 20 epochs
                    print('Training finished###########')
                    break
    writer.close()                                 
                
                
if __name__ == "__main__":
    bodypart = 'hip'
    dataset_name = 'Bone_0820_cls'
    img_folder = args.initial_path +'2D-slices/images'
    mask_folder = args.initial_path + '2D-slices/masks'
    train_img_list = args.initial_path + 'datalist_body_parts/img_list_12_12_train_' + bodypart + '_annotate_paired_2dslices.txt'
    val_img_list = args.initial_path + 'datalist_body_parts/img_list_12_12_val_' + bodypart + '_annotate_paired_2dslices.txt'
    dir_checkpoint = '2D-MobileSAM-onlyfusion-adapter_'+dataset_name+'_attentionpredicted'
    num_workers = 1
    if_vis = True
    epochs = 200
    
    label_mapping = args.initial_path  + 'segment_names_to_labels.pickle'
    train_dataset = MRI_dataset_multicls(args,img_folder, mask_folder, train_img_list,phase='train',targets=[args.targets],delete_empty_masks='subsample',label_mapping=label_mapping,if_prompt=False)
    eval_dataset = MRI_dataset_multicls(args,img_folder, mask_folder, val_img_list,phase='val',targets=[args.targets],delete_empty_masks='subsample',label_mapping=label_mapping,if_prompt=False)
    trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(eval_dataset, batch_size=16, shuffle=False, num_workers=num_workers)
    train_model(trainloader,valloader,dir_checkpoint,epochs)