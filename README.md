# Region Normalization for Image Inpainting

The paper can be found [here](https://arxiv.org/abs/1911.10375).

The codes are initial version, not the revised version. I will update these someday since I am busy currently. However, if you have any question about the paper/codes, you can contact me through Email(yutao666@mail.ustc.edu.cn).

Please run the codes where the python is Version 3.x and pytorch>=0.4.

## Preparation
Before running the codes, you should prepare training/evaluation image file list (flist) and mask file list (flist). You can refer to the folowing to generate .flist file:
```
python flist.py --path your_dataset_folder --output xxx.flist
```

## Training
There are some hyperparameters that you can adjust in the main.py. To train the model, you can run:
```
python main.py --bs 14 --gpus 2 --prefix rn --img_flist your_training_images.flist --mask_flist your_training_masks.flist
```
## Evaluation
To evaluate the model, you can run:
```
python eval.py --bs 32 --model your_checkpont_path --img_flist your_eval_images.flist --mask_flist your_eval_masks.flist
```

## Cite Us
```
@misc{yu2019region,
    title={Region Normalization for Image Inpainting},
    author={Tao Yu and Zongyu Guo and Xin Jin and Shilin Wu and Zhibo Chen and Weiping Li and Zhizheng Zhang and Sen Liu},
    year={2019},
    eprint={1911.10375},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Appreciation
The codes refer to [EdgeConnect](https://github.com/knazeri/edge-connect). Thanks for the authors of it！
