import os, torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import random
import torchio as tio
import slicerio
import nrrd
import monai
import pickle
import nibabel as nib
from scipy.ndimage import zoom
from monai.transforms import OneOf
import einops
from funcs import *
from torchvision.transforms import InterpolationMode
#from .utils.transforms import ResizeLongestSide


class MRI_dataset(Dataset):
    def __init__(self,args, img_folder, mask_folder, img_list,phase='train',sample_num=50,channel_num=1,crop=False,crop_size=1024,targets=['femur','hip'],part_list=['all'],cls=1,if_prompt=True,prompt_type='point',region_type='largest_15',prompt_num=15,delete_empty_masks=False,if_attention_map=None):
        super(MRI_dataset, self).__init__()
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.crop = crop
        self.crop_size = crop_size
        self.phase = phase
        self.channel_num=channel_num
        self.targets = targets
        self.segment_names_to_labels = []
        self.args = args
        self.cls = cls
        self.if_prompt = if_prompt
        self.region_type = region_type
        self.prompt_type = prompt_type
        self.prompt_num = prompt_num
        self.if_attention_map = if_attention_map
        
        for i,tag in enumerate(targets):
            self.segment_names_to_labels.append((tag,i))
            
        namefiles = open(img_list,'r')
        self.data_list = namefiles.read().split('\n')[:-1]

        if delete_empty_masks=='delete' or delete_empty_masks=='subsample':
            keep_idx = []
            for idx,data in enumerate(self.data_list):
                mask_path = data.split(' ')[1]
                if os.path.exists(os.path.join(self.mask_folder,mask_path)):
                    msk = Image.open(os.path.join(self.mask_folder,mask_path)).convert('L')
                else:
                    msk = Image.open(os.path.join(self.mask_folder.replace('2D-slices','2D-slices-generated'),mask_path)).convert('L')
                if 'all' in self.targets: # combine all targets as single target
                    mask_cls = np.array(np.array(msk,dtype=int)>0,dtype=int)
                else:
                    mask_cls = np.array(msk==self.cls,dtype=int)
                if part_list[0]=='all' and np.sum(mask_cls)>0:
                    keep_idx.append(idx) 
                elif np.sum(mask_cls)>0:
                    if_keep = False
                    for part in part_list:
                        if mask_path.find(part)>=0:
                            if_keep = True
                    if if_keep:
                        keep_idx.append(idx) 
            print('num with non-empty masks',len(keep_idx),'num with all masks',len(self.data_list))  
            if delete_empty_masks=='subsample':
                empty_idx = list(set(range(len(self.data_list)))-set(keep_idx))
                keep_empty_idx = random.sample(empty_idx, int(len(empty_idx)*0.1))
                keep_idx = empty_idx + keep_idx
            self.data_list = [self.data_list[i] for i in keep_idx] # keep the slices that contains target mask
  
        if phase == 'train':
            self.aug_img = [transforms.RandomEqualize(p=0.1),
                             transforms.ColorJitter(brightness=0.3, contrast=0.3,saturation=0.3,hue=0.3),
                             transforms.RandomAdjustSharpness(0.5, p=0.5),
                             ]
            self.transform_spatial = transforms.Compose([transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.2)),
                     transforms.RandomRotation(45)])
            transform_img = [transforms.ToTensor()]
        else:
            transform_img = [
                         transforms.ToTensor(),
                             ]
        self.transform_img = transforms.Compose(transform_img)
            
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self,index):
        # load image and the mask
        data = self.data_list[index]
        img_path = data.split(' ')[0]
        mask_path = data.split(' ')[1]
        slice_num = data.split(' ')[3] # total slice num for this object
        #print(img_path,mask_path)
        try:
            if os.path.exists(os.path.join(self.img_folder,img_path)):
                img = Image.open(os.path.join(self.img_folder,img_path)).convert('RGB')
            else:
                img = Image.open(os.path.join(self.img_folder.replace('2D-slices','2D-slices-generated'),img_path)).convert('RGB')
        except:
            # try to load image as numpy file
            img_arr = np.load(os.path.join(self.img_folder,img_path)) 
            img_arr = np.array((img_arr-img_arr.min())/(img_arr.max()-img_arr.min()+1e-8)*255,dtype=np.uint8)
            img_3c = np.tile(img_arr[:, :,None], [1, 1, 3])
            img = Image.fromarray(img_3c, 'RGB')
        if os.path.exists(os.path.join(self.mask_folder,mask_path)):
            msk = Image.open(os.path.join(self.mask_folder,mask_path)).convert('L')
        else:
            msk = Image.open(os.path.join(self.mask_folder.replace('2D-slices','2D-slices-generated'),mask_path)).convert('L')
                    
        if self.if_attention_map:
            slice_id = int(img_path.split('-')[-1].split('.')[0])
            slice_fraction = int(slice_id/int(slice_num)*4)
            img_id = '/'.join(img_path.split('-')[:-1]) +'_'+str(slice_fraction) + '.npy'
            attention_map = torch.tensor(np.load(os.path.join(self.if_attention_map,img_id)))
        else:
            attention_map = torch.zeros((64,64))
        
        img = transforms.Resize((self.args.image_size,self.args.image_size))(img)
        msk = transforms.Resize((self.args.image_size,self.args.image_size),InterpolationMode.NEAREST)(msk)
        
        state = torch.get_rng_state()
        if self.crop:
            im_w, im_h = img.size
            diff_w = max(0,self.crop_size-im_w)
            diff_h = max(0,self.crop_size-im_h)
            padding = (diff_w//2, diff_h//2, diff_w-diff_w//2, diff_h-diff_h//2)
            img = transforms.functional.pad(img, padding, 0, 'constant')
            torch.set_rng_state(state)
            t,l,h,w=transforms.RandomCrop.get_params(img,(self.crop_size,self.crop_size))
            img = transforms.functional.crop(img, t, l, h,w) 
            msk = transforms.functional.pad(msk, padding, 0, 'constant')
            msk = transforms.functional.crop(msk, t, l, h,w)
        if self.phase =='train':
            # add random optimazition
            aug_img_fuc = transforms.RandomChoice(self.aug_img)
            img = aug_img_fuc(img)

        img = self.transform_img(img)
        if self.phase == 'train':
            # It will randomly choose one
            random_transform = OneOf([monai.transforms.RandGaussianNoise(prob=0.5, mean=0.0, std=0.1),\
                                      monai.transforms.RandKSpaceSpikeNoise(prob=0.5, intensity_range=None, channel_wise=True),\
                                      monai.transforms.RandBiasField(degree=3),\
                                      monai.transforms.RandGibbsNoise(prob=0.5, alpha=(0.0, 1.0))
                                     ],weights=[0.3,0.3,0.2,0.2])
            img = random_transform(img).as_tensor()
        else:
            if img.mean()<0.05:
                img = min_max_normalize(img)
                img = monai.transforms.AdjustContrast(gamma=0.8)(img)

        
        if 'all' in self.targets: # combine all targets as single target
            msk = np.array(np.array(msk,dtype=int)>0,dtype=int)
        else:
            msk = np.array(msk,dtype=int)
            
        mask_cls = np.array(msk==self.cls,dtype=int)

        if self.phase=='train' and (not self.if_attention_map==None):
            mask_cls = np.repeat(mask_cls[np.newaxis,:, :], 3, axis=0)
            both_targets = torch.cat((img.unsqueeze(0), torch.tensor(mask_cls).unsqueeze(0)),0)
            transformed_targets = self.transform_spatial(both_targets)
            img = transformed_targets[0]
            mask_cls = np.array(transformed_targets[1][0].detach(),dtype=int)
        
        img = (img-img.min())/(img.max()-img.min()+1e-8)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        
        # generate mask and prompt
        if self.if_prompt:
            if self.prompt_type =='point':
                prompt,mask_now = get_first_prompt(mask_cls,region_type=self.region_type,prompt_num=self.prompt_num)
                pc = torch.as_tensor(prompt[:,:2], dtype=torch.float)
                pl = torch.as_tensor(prompt[:, -1], dtype=torch.float)
                msk = torch.unsqueeze(torch.tensor(mask_now,dtype=torch.long),0)
                return {'image':img,
                    'mask':msk,
                    'point_coords': pc,
                    'point_labels':pl,
                    'img_name':img_path,
                    'atten_map':attention_map,
            }
            elif self.prompt_type =='box':
                prompt,mask_now = get_top_boxes(mask_cls,region_type=self.region_type,prompt_num=self.prompt_num)
                box = torch.as_tensor(prompt, dtype=torch.float)
                msk = torch.unsqueeze(torch.tensor(mask_now,dtype=torch.long),0)
                return {'image':img,
                    'mask':msk,
                    'boxes':box,
                    'img_name':img_path,
                    'atten_map':attention_map,
            }
        else:
            msk = torch.unsqueeze(torch.tensor(mask_cls,dtype=torch.long),0)
            return {'image':img,
                'mask':msk,
                'img_name':img_path,
                'atten_map':attention_map,
        }