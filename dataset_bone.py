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


class MRI_dataset_multicls(Dataset):
    def __init__(self, args, img_folder, mask_folder, img_list, phase='train', sample_num=50, channel_num=1,
                 crop=False, crop_size=1024, targets=['combine_all'], part_list=['all'], if_prompt=True, 
                 prompt_type='point', if_spatial = True, region_type='largest_20', prompt_num=20, delete_empty_masks=False, 
                 label_mapping=None, reference_slice_num=0, if_attention_map=None,label_frequency_path=None):
        super(MRI_dataset_multicls, self).__init__()
        self.initialize_parameters(args, img_folder, mask_folder, img_list, phase, sample_num, channel_num,
                                   crop, crop_size, targets, part_list, if_prompt, prompt_type, if_spatial, region_type,
                                   prompt_num, delete_empty_masks, label_mapping, reference_slice_num, if_attention_map,label_frequency_path)
        self.load_label_mapping()
        self.prepare_data_list()
        self.filter_data_list()
        if phase == 'train':
            self.setup_transformations_train(crop_size)
        else:
            self.setup_transformations_other()

    def initialize_parameters(self, args, img_folder, mask_folder, img_list, phase, sample_num, channel_num,
                              crop, crop_size, targets, part_list, if_prompt, prompt_type, if_spatial, region_type,
                              prompt_num, delete_empty_masks, label_mapping, reference_slice_num, if_attention_map,label_frequency_path):
        self.args = args
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.img_list = img_list
        self.phase = phase
        self.sample_num = sample_num
        self.channel_num = channel_num
        self.crop = crop
        self.crop_size = crop_size
        self.targets = targets
        self.part_list = part_list
        self.if_prompt = if_prompt
        self.prompt_type = prompt_type
        self.if_spatial = if_spatial
        self.region_type = region_type
        self.prompt_num = prompt_num
        self.delete_empty_masks = delete_empty_masks
        self.label_mapping = label_mapping
        self.reference_slice_num = reference_slice_num
        self.if_attention_map = if_attention_map
        self.label_dic = {}
        self.label_frequency_path = label_frequency_path

    def load_label_mapping(self):
        # Load the basic label mappings from a pickle file
        if self.label_mapping:
            with open(self.label_mapping, 'rb') as handle:
                self.segment_names_to_labels = pickle.load(handle)
            self.label_dic = {seg[1]: seg[0] for seg in self.segment_names_to_labels}
            self.label_name_list = [seg[0] for seg in self.segment_names_to_labels]
            print(self.label_dic)
        else:
            self.label_dic = {value: 'all' for value in range(1, 256)}
        
        # Load frequency data and remap classes if required
        if 'remap_frequency' in self.targets:
            self.load_and_remap_classes_based_on_frequency()

    def load_and_remap_classes_based_on_frequency(self):
        if self.label_frequency_path:
            with open(self.label_frequency_path, 'r') as file:
                all_label_frequencies = json.load(file)
            all_label_frequencies = all_label_frequencies['train']
            
            
            # Example to select the target region dynamically based on some condition or configuration
            target_region = self.part_list[0] 
            if target_region in all_label_frequencies:
                label_frequencies = all_label_frequencies[target_region]
                self.label_frequencies = label_frequencies
                #print(label_frequencies)
                self.remap_classes_based_on_frequency(label_frequencies)
            else:
                print(f"Warning: No frequency data found for the target region '{target_region}'. No remapping applied.")
    
    def remap_classes_based_on_frequency(self, label_frequencies):
        # Determine the frequency threshold for high vs. low frequency classes
        total = max(label_frequencies.values())
        high_freq_threshold = total * 0.5  # Adjust this threshold as needed
        
        # Initialize dictionaries to hold new class mappings
        high_freq_classes = {}
        low_freq_classes = {}
        
        # Assign classes to high or low frequency based on the threshold
        for label, freq in label_frequencies.items():
            if freq >= high_freq_threshold:
                high_freq_classes[label] = freq
            else:
                low_freq_classes[label] = freq
    
        # Update label dictionary based on the frequency classification
        #self.label_dic: {old_cls: old_name}
        new_label_dic = {}
        for cls, name in self.label_dic.items():
            if name in high_freq_classes:
                new_label_dic[cls] = name  # Retain original name for high frequency classes
            elif name in low_freq_classes:
                new_label_dic[cls] = 'combined_low_freq'  # Combine low frequency classes into one
    
        self.updated_label_dic = new_label_dic
        #new_label_dic: {old_cls: new_name}
        #print("Updated label dictionary with frequency remapping:", new_label_dic)
        
        #print('new_label_dic:',new_label_dic)
    
        # Sort high frequency keys by their frequency in descending order
        sorted_high_freq_labels = sorted(high_freq_classes.items(), key=lambda item: item[1], reverse=True)
        
        # Create a mapping for high frequency classes based on the sorted order
        original_to_new = {label: idx + 1 for idx, (label, _) in enumerate(sorted_high_freq_labels)}

        
        combined_low_freq_class_id = len(original_to_new) + 1
        # Ensure combined low frequency class is mapped correctly
        if 'combined_low_freq' in new_label_dic.values():
            for cls in low_freq_classes.keys():
                original_to_new[cls] = combined_low_freq_class_id
                
        # orignal_to_new {old_name:new_cls} 
        #print('original_to_new:',original_to_new)

        
        # Create additional dictionaries
        self.old_name_to_new_name = {self.label_dic[cls]: new_label for cls, new_label in new_label_dic.items()}
        self.old_cls_to_new_cls = {cls: original_to_new[self.label_dic[cls]] for cls in self.label_dic.keys() if self.label_dic[cls] in original_to_new}

        print('remapped label dic:',self.old_name_to_new_name)
        print('remapped cls dic:',self.old_cls_to_new_cls)
            
    def prepare_data_list(self):
        with open(self.img_list, 'r') as namefiles:
            self.data_list = namefiles.read().split('\n')[:-1]
        self.sp_symbol = ',' if ',' in self.data_list[0] else ' '

    def filter_data_list(self):
        keep_idx = []
        for idx, data in enumerate(self.data_list):
            img_path, mask_path = self.extract_paths(data)
            msk = Image.open(os.path.join(self.mask_folder, mask_path)).convert('L')
            mask_cls = self.determine_mask_class(msk)

            if self.should_keep(mask_cls, mask_path):
                keep_idx.append(idx)
                if self.reference_slice_num > 1:
                    self.add_reference_slice(img_path, mask_path, data)

        self.data_list = [self.data_list[i] for i in keep_idx]
        print('num with non-empty masks', len(keep_idx), 'num with all masks', len(self.data_list))

    def extract_paths(self, data):
        img_path = data.split(self.sp_symbol)[0]
        mask_path = data.split(self.sp_symbol)[1]
        return img_path.lstrip('/'), mask_path.lstrip('/')

    def determine_mask_class(self, msk):
        if 'combine_all' in self.targets:
            return np.array(msk, dtype=int) > 0
        elif self.targets[0] in self.label_name_list:
            return np.array(msk, dtype=int) == self.cls
        return np.array(msk, dtype=int)

    def should_keep(self, mask_cls, mask_path):
        if self.delete_empty_masks:
            has_mask = np.any(mask_cls > 0)
            if has_mask:
                if self.part_list[0] == 'all':
                    return True
                return any(mask_path.find(part) >= 0 for part in self.part_list)
            return False
        return True


    def add_reference_slice(self, img_path, mask_path, data):
        volume_name = ''.join(img_path.split('-')[:-1])  # get volume name
        slice_num = data.split(self.sp_symbol)[2]
        if volume_name not in self.reference_slices:
            self.reference_slices[volume_name] = []
        self.reference_slices[volume_name].append((img_path, mask_path, slice_num))

    def setup_transformations_train(self, crop_size):
        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.aug_img = transforms.RandomChoice([
            transforms.RandomEqualize(p=0.1),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.RandomAdjustSharpness(0.5, p=0.5),
        ])
        if self.if_spatial:
                self.transform_spatial = transforms.Compose([transforms.RandomResizedCrop(self.crop_size, scale=(0.5, 1.5), interpolation=InterpolationMode.NEAREST),
                         transforms.RandomRotation(45, interpolation=InterpolationMode.NEAREST)])

    def setup_transformations_other(self):
        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # Load image and mask, handle missing files
        data = self.data_list[index]
        img, msk, img_path, mask_path, slice_num = self.load_image_and_mask(data)
        
        # Optional: Load attention map
        attention_map = self.load_attention_map(img_path, slice_num) if self.if_attention_map else torch.zeros((64, 64))
    
        # Handle reference slices if necessary
        if self.reference_slice_num > 1:
            img, msk = self.handle_reference_slices(img_path, mask_path, slice_num)
        
        # Apply transformations
        img, msk = self.apply_transformations(img, msk)
    
        # Generate and process masks and prompts
        output_dict = self.prepare_output(img, msk, img_path, mask_path,attention_map)
        
    
        return output_dict
        
    def load_image_and_mask(self, data):
        img_path, mask_path = self.extract_paths(data)
        slice_num = data.split(self.sp_symbol)[3]  # Extract total slice number for this object
        
        img_folder = self.img_folder
        msk_folder = self.mask_folder
        
        img = Image.open(os.path.join(img_folder, img_path)).convert('RGB')
        msk = Image.open(os.path.join(msk_folder, mask_path)).convert('L')
    
        # Resize images for processing
        img = transforms.Resize((self.args.image_size, self.args.image_size))(img)
        msk = transforms.Resize((self.args.image_size, self.args.image_size), InterpolationMode.NEAREST)(msk)
    
        return img, msk, img_path, mask_path, int(slice_num)

    def load_attention_map(self, img_path, slice_num):
        slice_id = int(img_path.split('-')[-1].split('.')[0])
        slice_fraction = int(slice_id / slice_num * 4)
        img_id = '/'.join(img_path.split('-')[:-1]) + '_' + str(slice_fraction) + '.npy'
        attention_map = torch.tensor(np.load(os.path.join(self.if_attention_map, img_id)))
        return attention_map


    def apply_crop(self, img, msk):
        im_w, im_h = img.size
        diff_w = max(0, self.crop_size - im_w)
        diff_h = max(0, self.crop_size - im_h)
        padding = (diff_w // 2, diff_h // 2, diff_w - diff_w // 2, diff_h - diff_h // 2)
        img = transforms.functional.pad(img, padding, 0, 'constant')
        msk = transforms.functional.pad(msk, padding, 0, 'constant')
        t, l, h, w = transforms.RandomCrop.get_params(img, (self.crop_size, self.crop_size))
        img = transforms.functional.crop(img, t, l, h, w)
        msk = transforms.functional.crop(msk, t, l, h, w)
        return img, msk
        
    def apply_transformations(self, img, msk):
        if self.crop:
            img, msk = self.apply_crop(img, msk)
        if self.phase == 'train':
            img = self.aug_img(img)
        img = self.transform_img(img)
        if self.phase =='train' and self.if_spatial:
            mask_cls = np.array(msk,dtype=int)
            mask_cls = np.repeat(mask_cls[np.newaxis,:, :], 3, axis=0)
            both_targets = torch.cat((img.unsqueeze(0), torch.tensor(mask_cls).unsqueeze(0)),0)
            transformed_targets = self.transform_spatial(both_targets)
            img = transformed_targets[0]
            mask_cls = np.array(transformed_targets[1][0].detach(),dtype=int)
            msk = torch.tensor(mask_cls)
        return img, msk

    def handle_reference_slices(self, img_path, mask_path, slice_num):
        volume_name = ''.join(img_path.split('-')[:-1])
        ref_slices, ref_msks = [], []
        reference_slices = self.reference_slices.get(volume_name, [])
        for ref_slice in reference_slices:
            ref_img_path, ref_msk_path, _ = ref_slice
            ref_img = Image.open(os.path.join(self.img_folder, ref_img_path)).convert('RGB')
            ref_img = transforms.Resize((self.args.image_size, self.args.image_size))(ref_img)
            ref_img = self.transform_img(ref_img)
            ref_img = torch.unsqueeze(ref_img, 0)
            
            ref_msk = Image.open(os.path.join(self.mask_folder, ref_msk_path)).convert('L')
            ref_msk = transforms.Resize((self.args.image_size, self.args.image_size), InterpolationMode.NEAREST)(ref_msk)
            ref_msk = torch.tensor(ref_msk, dtype=torch.long)
            ref_msks.append(torch.unsqueeze(ref_msk, 0))
    
        img = torch.cat(ref_slices, dim=0)
        msk = torch.cat(ref_msks, dim=0)
        return img, msk
    
    def remap_classes_sequentially(self, mask, label_frequencies):
        # Apply the mapping to the mask
        remapped_mask = mask.copy()
        for old_cls, new_cls in  self.old_cls_to_new_cls.items():
            remapped_mask[mask == old_cls] = new_cls
        return remapped_mask


    def prepare_output(self, img, msk, img_path, mask_path, attention_map):
        # Normalize the image
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    
        msk = np.array(msk, dtype=int)
        #print('ori_msk:',np.unique(msk))
        if self.label_frequency_path:
            msk = self.remap_classes_sequentially(msk,self.label_frequencies)  # Assuming msk is already using updated IDs
            #print('new_msk------------------------:',self.old_cls_to_new_cls)
            # Prepare one-hot encoding for the remapped classes
        
        unique_classes = np.unique(msk).tolist()
        if 0 in unique_classes:
            unique_classes.remove(0)
    
        if len(unique_classes) > 0:
            selected_dic = {k: self.label_dic[k] for k in unique_classes if k in self.label_dic}
        else:
            selected_dic = {}
    
        if self.targets[0] == 'random':
            mask_cls, selected_label, cls_one_hot = self.handle_random_target(msk, unique_classes, selected_dic)
        elif self.targets[0] in self.label_name_list:
            selected_label = self.targets[0]
            mask_cls = np.array(msk == self.cls, dtype=int)
            cls_one_hot = torch.zeros(len(self.label_dic), dtype=torch.long)
            cls_one_hot[self.cls - 1] = 1
        else:
            selected_label = self.targets[0]
            mask_cls = msk
            cls_one_hot = torch.zeros(len(self.label_dic), dtype=torch.long)
    
        # Handle prompts
        if self.if_prompt:
            prompt, mask_now, mask_cls = self.generate_prompt(mask_cls)
            ref_msk,_ = torch.max(mask_now>0,dim=0)
            return_dict = {'image': img, 'mask': mask_now, 'selected_label_name': selected_label,
                           'cls_one_hot': cls_one_hot, 'prompt': prompt, 'img_name': img_path,
                           'mask_ori': msk, 'mask_cls': mask_cls, 'all_label_dic': selected_dic,'ref_mask':ref_msk}
        else:
            if len(mask_cls.shape)==2:
                msk = torch.unsqueeze(torch.tensor(mask_cls,dtype=torch.long),0)
            elif len(mask_cls.shape)==4:
                msk = torch.squeeze(torch.tensor(mask_cls,dtype=torch.long))
            else:
                msk = torch.tensor(mask_cls,dtype=torch.long)
            ref_msk,_ = torch.max(msk>0,dim=0)
            #print('unique mask values:',msk.unique())
            return_dict = {'image': img, 'mask': msk, 'selected_label_name': selected_label,
                           'cls_one_hot': cls_one_hot, 'img_name': img_path, 'mask_ori': msk,'ref_mask':ref_msk}
    
        return return_dict
        
    def generate_prompt(self, mask_cls):
        if self.prompt_type == 'point':
            prompt, mask_now = get_first_prompt(mask_cls, region_type=self.region_type, prompt_num=self.prompt_num)
        elif self.prompt_type == 'box':
            prompt, mask_now = get_top_boxes(mask_cls, region_type=self.region_type, prompt_num=self.prompt_num)
        else:
            prompt = mask_now = None
        
        # Handling the shape of mask_now for return
        if mask_now is not None:
            if len(mask_now.shape) == 2:
                mask_now = torch.unsqueeze(torch.tensor(mask_now, dtype=torch.long), 0)
                mask_cls = torch.unsqueeze(torch.tensor(mask_cls, dtype=torch.long), 0)
            elif len(mask_now.shape) == 4:
                mask_now = torch.squeeze(torch.tensor(mask_now, dtype=torch.long))
            else:
                mask_now = torch.tensor(mask_now, dtype=torch.long)
                mask_cls = torch.tensor(mask_cls, dtype=torch.long)
    
        return prompt, mask_now, mask_cls


    def handle_random_target(self, msk, unique_classes, selected_dic):
        if len(unique_classes) > 0:
            random_selected_cls = random.choice(unique_classes)
            selected_label = selected_dic[random_selected_cls]
            mask_cls = np.array(msk == random_selected_cls, dtype=int)
            
            cls_one_hot = torch.zeros(len(self.label_dic), dtype=torch.long)
            cls_one_hot[random_selected_cls - 1] = 1
        else:
            selected_label = None
            mask_cls = torch.zeros_like(msk)  # assuming msk is already a numpy array
            cls_one_hot = torch.zeros(len(self.label_dic), dtype=torch.long)
    
        return mask_cls, selected_label, cls_one_hot