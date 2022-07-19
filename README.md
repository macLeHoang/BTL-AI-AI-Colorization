# AI COLORIZATION

## Weights:
- Pre-trained weights in 20 epochs: [link](https://drive.google.com/drive/u/0/folders/1ta_xpQ0l4mKbwxMrerxp216NLDGowbjw)
- GAN train on 71 epochs: [link](https://drive.google.com/drive/u/0/folders/1ta_xpQ0l4mKbwxMrerxp216NLDGowbjw)

## Some exmaples:
![](https://github.com/macLeHoang/BTL-AI-AI-Colorization/blob/main/examples/exResult.jpg?raw=true)
Left: Origin - Middle: Gray - Right: Colored

## Evaluation:
### AuC
Test on COCO test 2017

<img src= "https://github.com/macLeHoang/BTL_AI-Colorization/blob/main/examples/per_img_75_0.01.png" width="330" height="221" /><img src= "https://github.com/macLeHoang/BTL_AI-Colorization/blob/main/examples/per_pixel_75_0.01.png" width="330" height="221" /> 

### Classfication task
Test on ImageNet_V2 which label space is the same as ImageNet-1k. Using VGG19  as classification model

Type | Top 1 - accuracy
--- | --- |
Origin | 51.73%
Gray | 38.68%
Colored | 41.28%
