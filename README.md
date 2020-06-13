# Region Normalization for Image Inpainting

The paper can be found [here](https://arxiv.org/abs/1911.10375).

The codes are initial version, not the revised version. I will update these someday since I am busy currently. However, if you have any question about the paper/codes, you can contact me through Email(yutao666@mail.ustc.edu.cn).

Please run the codes where the python is Version 3.x and pytorch>=0.4.

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
@inproceedings{DBLP:conf/aaai/YuGJW0LZL20,
  author    = {Tao Yu and
               Zongyu Guo and
               Xin Jin and
               Shilin Wu and
               Zhibo Chen and
               Weiping Li and
               Zhizheng Zhang and
               Sen Liu},
  title     = {Region Normalization for Image Inpainting},
  booktitle = {The Thirty-Fourth {AAAI} Conference on Artificial Intelligence, {AAAI}
               2020, The Thirty-Second Innovative Applications of Artificial Intelligence
               Conference, {IAAI} 2020, The Tenth {AAAI} Symposium on Educational
               Advances in Artificial Intelligence, {EAAI} 2020, New York, NY, USA,
               February 7-12, 2020},
  pages     = {12733--12740},
  publisher = {{AAAI} Press},
  year      = {2020},
  url       = {https://aaai.org/ojs/index.php/AAAI/article/view/6967},
  timestamp = {Wed, 03 Jun 2020 18:42:45 +0200},
  biburl    = {https://dblp.org/rec/conf/aaai/YuGJW0LZL20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}


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
