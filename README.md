# AI COLORIZATION
## Weights:
- Pre-trained weights in 20 epochs: [link](https://drive.google.com/drive/u/0/folders/1ta_xpQ0l4mKbwxMrerxp216NLDGowbjw)
- GAN train on 71 epochs: [link](https://drive.google.com/drive/u/0/folders/1ta_xpQ0l4mKbwxMrerxp216NLDGowbjw)

## Some exmaples:
![](https://github.com/macLeHoang/BTL-AI-AI-Colorization/blob/main/examples/exResult.jpg?raw=true)
Left: Origin - Middle: Gray - Right: Colored


## Run Example
```bash
git clone https://github.com/macLeHoang/BTL_AI-Colorization
cd BTL_AI-Colorization

python run.py \
-k \
-w [your path to weight file] \
-i [your path to image]
-st [path to store gen_img - you can delete this line]
```
or this for specified dims
```
git clone https://github.com/macLeHoang/BTL_AI-Colorization
cd BTL_AI-Colorization

python BTL_AI-Colorization/run.py \
-s ([w], [h])
-w [your path to weight file] \
-i [your path to image]
-st [path to store gen_img - you can delete this line]
```

## Evaluation:
### AuC
Calculate histogram cumulative of per pixel RMS error and of image RMS error on COCO test 2017

<img src= "https://github.com/macLeHoang/BTL_AI-Colorization/blob/main/examples/per_img_75_0.01.png" width="330" height="221" /><img src= "https://github.com/macLeHoang/BTL_AI-Colorization/blob/main/examples/per_pixel_75_0.01.png" width="330" height="221" /> 

### Classfication task
Do classification task on ImageNet_V2 which label space is the same as ImageNet-1k. Using VGG19 as classification model. 
First do task on origin image, then wash out all color and finally on re-colored. 

Type | Top 1 - accuracy
--- | --- |
Origin | 51.73%
Gray | 38.68%
Colored | 41.28%
