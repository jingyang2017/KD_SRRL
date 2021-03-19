# 
## paper
[Knowledge distillation via softmax regression representation learning(ICLR2021)](https://openreview.net/pdf?id=ZzwDy_wiWv)
Jing Yang, Brais Martinez, Adrian Bulat, Georgios Tzimiropoulos

## Requirements
- Python >= 3.6
- PyTorch >= 1.0.1

## ImageNet Training
```python train_imagenet_distillation.py --net_s resnet18S --net_t resnet34T ```
```python train_imagenet_distillation.py --net_s MobileNet --net_t resnet50T ```

## Citation
```
@inproceedings{kd_srrl, 
  title={Knowledge distillation via softmax regression representation learning},
  author={Jing Yang, Brais Martinez, Adrian Bulat, Georgios Tzimiropoulos},
  booktitle={ICLR2021},
  year={2021}  
}
```

## License
This project is licensed under the MIT License



