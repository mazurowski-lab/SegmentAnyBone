# SegmentAnyBone: A Universal Model that Segments Any Bone at Any Location on MRI

[![arXiv Paper](https://img.shields.io/badge/arXiv-2401.12974-orange.svg?style=flat)](https://arxiv.org/abs/2401.12974)


[SegmentAnyBone](https://arxiv.org/abs/2401.12974) is a foundational model-based bone segmentation algorithm adapted from [Segment Anything Model (SAM)](https://pages.github.com/](https://github.com/facebookresearch/segment-anything)https://github.com/facebookresearch/segment-anything) for MRI scans. It can segment bones in the following **17 body parts**:

**Warning:** Please note that this software is developed for research purposes and is not intended for clinical use yet. Users should exercise caution and are advised against employing it immediately in clinical or medical settings.

**`Humerus`**  |  **`Thoracic Spine`**   |  **`Lumbar Spine`**   | **`Forearm`** | **`Pelvis`** |  **`Hand`** |  **`Lower Leg`** 

 **`Shoulder`** | **`Chest`**  |  **`Arm`**   |  **`Elbow`**   | **`Hip`** | **`Wrist`** |  **`Thigh`** |  **`Knee`** |  **`Foot`** |  **`Ankle`** 

![Screenshot](segment-any-bone.png)

## Dataset

![Screenshot](dataset.png)

## Updates [Dec-23-2025]: 
We update the codes to fine-tune the SegmentAnyBone to custom datasets, and the dataset preparation guideline can be found in our other repo: ![finetume-sam](https://github.com/mazurowski-lab/finetune-SAM/tree/main).
- you can further fine-tune SegmentAnyBone to your dataset.
- you can further extend it into multi-class classification.
- [todo] this version might be unstable and we are still working on different testing.
- [todo] We will be sharing two datasets with annotations soon. 

## Installation & Usage

You can clone the repository and install required Python packages by running following commands:
```
git clone https://github.com/mazurowski-lab/SegmentAnyBone.git
cd SegmentAnybone; pip install -r requirements.txt
```

## Sample Output

![Screenshot](sample_output.png)

### Model Checkpoints

You can download required model checkpoints from following links:
[**`Mobile SAM`**](https://github.com/ChaoningZhang/MobileSAM/tree/master/weights)
[**`SegmentAnyBone`**](https://drive.google.com/drive/folders/1PGKXlhj8b-fFEkYVw-Cmpj8qLSrJmTEO?usp=sharing)

After cloning the repository and downloading the checkpoints to the project folder, you should put your 3D MRI volume in `/images`, and your ground truth mask in `/masks` folder if you want to evaluate the segmentation performance of SegmentAnyBone. If you need to segment 3D volume instead of 1 slice you can see the 3D segmentation mask under `/predicted_masks` after you run **_predictVolume()_** or **_predictAndEvaluateVolume()_** function. [This notebook](demo.ipynb)  will guide you to use SegmentAnyBone in a slice-based and volume-based manner thanks to following **_predictSlice()_**, **evaluateSlicePrediction()_**, and **_predictAndEvaluateVolume()_** functions: 

```python
ori_img, predictedSliceMask, atten_map = predictSlice(
    image_name = '2.nii.gz', 
    lower_percentile = 1,
    upper_percentile = 99,
    slice_id = 50, # slice number
    attention_enabled = True, # if you want to use the depth attention
)

msk_gt, dsc_gt = evaluateSlicePrediction(
    mask_pred = predictedSliceMask, 
    mask_name = '2.nrrd', 
    slice_id = 50
)
```

```python
mask = predictVolume(
    image_name = '2.nii.gz', 
    lower_percentile = 1, 
    upper_percentile = 99
)

predictAndEvaluateVolume(
    image_name = '2.nii.gz', 
    mask_name = '2.nrrd',
    lower_percentile = 1, 
    upper_percentile = 99
)
```
## License

The model is licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

## Citation

If you find our work to be useful for your research, please cite [our paper](https://arxiv.org/abs/2401.12974):

```bibtex
@misc{gu2024segmentanybone,
      title={SegmentAnyBone: A Universal Model that Segments Any Bone at Any Location on MRI}, 
      author={Hanxue Gu and Roy Colglazier and Haoyu Dong and Jikai Zhang and Yaqian Chen and Zafer Yildiz and Yuwen Chen and Lin Li and Jichen Yang and Jay Willhite and Alex M. Meyer and Brian Guo and Yashvi Atul Shah and Emily Luo and Shipra Rajput and Sally Kuehn and Clark Bulleit and Kevin A. Wu and Jisoo Lee and Brandon Ramirez and Darui Lu and Jay M. Levin and Maciej A. Mazurowski},
      year={2024},
      eprint={2401.12974},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
```
}
