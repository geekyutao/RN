# Region Normalization for Image Inpainting

The paper can be found [here](https://arxiv.org/abs/1911.10375). If you have any question about the paper/codes, you can contact me through Email(yutao666@mail.ustc.edu.cn).

Please run the codes where the python is Version 3.x and pytorch>=0.4.

PS: 1) The results of this version codes are better than those in the paper. The original base inpainting model that RN uses is not very stable (the result variance is a bit large) and we only reported conservative results. However, we optimized the base model and improved its robustness after the pulication so that the results now are better. 2) RN wants to bring an insight that spatially region-wise normalization is better for some CV tasks such as inpainting. Theoretically, RN can be both BN-style or IN-style. Both have pros and cons. IN-style RN gives less blurring results and achieves style consistence to background in some extent, while suffers from spatial inconsistence if the model representation ability is limited. BN-style RN gives higher PSNR on an aligned validation data, but makes regions more blurring and causes much data-bias risk when testing data distribution has a certain shift to training data distribution. One chooses the RN style according to the specific scene. (See [issue #12](https://github.com/geekyutao/RN/issues/12))

## Repo Update:
- [04/26/2022] Support torch >= 1.7; fix old-version issues.


## Preparation
Before running the codes, you should prepare training/evaluation image file list (flist) and mask file list (flist). You can refer to the folowing command to generate .flist file:
```
python flist.py --path your_dataset_folder --output xxx.flist
```

## Training
There are some hyperparameters that you can adjust in the main.py. To train the model, you can run:
```
python main.py --bs 14 --gpus 2 --prefix rn --img_flist your_training_images.flist --mask_flist your_training_masks.flist
```
PS: You can set the "--bs" and "--gpus" to any number as you like. The above is just an example.

## Evaluation
To evaluate the model, you can use GPU or CPU to run.

For GPU:
```
python eval.py --bs your_batch_size --model your_checkpoint_path --img_flist your_eval_images.flist --mask_flist your_eval_masks.flist
```

For CPU:
```
python eval.py --cpu --bs your_batch_size --model your_checkpoint_path --img_flist your_eval_images.flist --mask_flist your_eval_masks.flist
```

PS: The pretrained model under folder './pretrained_model/' is trained from Places2 dataset with [Irregular Mask](https://nv-adlr.github.io/publication/partialconv-inpainting) dataset. **Please train RN from scratch if you test data not from Places2 or using regular mask.**

## Cite Us
Please cite us if you find this work helps.

```
@inproceedings{yu2020region,
  title={Region Normalization for Image Inpainting.},
  author={Yu, Tao and Guo, Zongyu and Jin, Xin and Wu, Shilin and Chen, Zhibo and Li, Weiping and Zhang, Zhizheng and Liu, Sen},
  booktitle={AAAI},
  pages={12733--12740},
  year={2020}
}
```

## Appreciation
The codes refer to [EdgeConnect](https://github.com/knazeri/edge-connect). Thanks for the authors of it！
