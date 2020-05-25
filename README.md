# PSGAN

Code for our CVPR 2020 **oral** paper "[PSGAN: Pose and Expression Robust Spatial-Aware GAN for Customizable Makeup Transfer](https://arxiv.org/abs/1909.06956)".

Contributed by [Wentao Jiang](https://wtjiang98.github.io), [Si Liu](http://colalab.org/people), Chen Gao, Jie Cao, Ran He, [Jiashi Feng](https://sites.google.com/site/jshfeng/), [Shuicheng Yan](https://www.ece.nus.edu.sg/stfpage/eleyans/)

![](psgan_framework.png)

## Checklist
- [x] more results 
- [ ] video demos
- [ ] partial makeup transfer example
- [ ] interpolated makeup transfer example
- [ ] inference on GPU
- [ ] training code


## Requirements
   The code was tested on Ubuntu 16.04, with Python 3.6 and PyTorch 1.0.

## Test

1. modify the image files for makeup transfer in ``makeup/main.py`` (optional)
2. run ``makeup/main.py``
  
## More Results

#### MT-Dataset (frontal face images with neutral expression)

![](MT-results.png)


#### MWild-Dataset (images with different poses and expressions)

![](MWild-results.png)

#### Video Makeup Transfer (by simply applying PSGAN on each frame)

![](Video_MT.png)

## Citation
Please consider citing this project in your publications if it helps your research. The following is a BibTeX reference. The BibTeX entry requires the url LaTeX package.

~~~
@inproceedings{jiang2019psgan,
  title={PSGAN: Pose and Expression Robust Spatial-Aware GAN for Customizable Makeup Transfer},
  author={Jiang, Wentao and Liu, Si and Gao, Chen and Cao, Jie and He, Ran and Feng, Jiashi and Yan, Shuicheng},
  booktitle={CVPR},
  year={2020}
}
~~~

## Acknowledge
Some of the codes are built upon [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) and [BeautyGAN](https://github.com/wtjiang98/BeautyGAN_pytorch). 