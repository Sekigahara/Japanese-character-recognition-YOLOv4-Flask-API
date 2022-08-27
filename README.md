# Japanese character recognition Flask REST API with YOLOv4
This repositories is adapting YOLOv4 to Flask API with Darkeras [here](https://github.com/tranleanh/darkeras-yolov4).
The client to this API is targeted to be Java Android using retrofit.

## Main Repositories
- High level and low level overview of the project can be found in the main repositories.
- Main respositories can be found [here](https://github.com/Sekigahara/Multilabel-classification-Japanese-character-with-YOLOv4).

## Installation

<b>This API runs on Python 3.8</b>

<b>Weight placement</b>
- Place your trained weight and obj.names to weight folder

<b>Darkeras instalation and various changes are necessarily required inorder to make the API runs</b>
- After cloning or download this repo, clone Darkeras repo [here](https://github.com/tranleanh/darkeras-yolov4) and changes the directory name from <b> darkeras-yolov4 </b> to <b>darkeras_yolov4</b>.
- Execute this with pip to install the requirement of darkeras and API :
```
pip install -r darkeras_yolov4/requirements.txt
``` 
- Create empty ```__init__.py``` files and place it in ```main directory, darkeras_yolov4 and darkeras_yolov4/core_yolov4```
- In ```darkeras_yolov4/core_yolov4/config.py``` at line 14 change it to the ```weight/obj.names```
- In ```darkeras_yolov4/core_yolov4/utils.py``` at line 6 change it to ```from darkeras_yolov4.core_yolov4.config import cfg```
- In ```darkeras_yolov4/core_yolov4/utils.py``` at line 77 add new parameter ```encoding="utf8``` inside open function

<b> API Setup </b>
- Install API requirements using this : 
```
pip install -r requirements.txt
```
- Download arial-unicode-ms.ttf [here](https://github.com/texttechnologylab/DHd2019BoA/blob/master/fonts/Arial%20Unicode%20MS.TTF) or any font that might supported for unicode font(to change the font you can change it in models.py at line 218).
- (Optional) You can try this API with default configuration and trained hiragana weight [here](https://drive.google.com/file/d/1kJ_9nSmBp_qPFG4I6r_fTim40bcVkmuT/view?usp=sharing)
- Make sure to change the weight filename in API.py. For example : ```weight/[your_weight_name.weights]```
- Make sure to change the input size and num class in API.py based on your preference when training. ```(Default. input size:736 and num class:46)```
- Make sure to adjust your IP or Port preference before run the API.
- You can change the IOU threshold and NMS to see which one works better.
- After everything is installed you can start the API with
```
python API.py
```

## Request
- The request endpoint is using(POST request) ```http://ip:port/api/detect``` <br>
Resize is optional from client, however you need uncomment line 31 for resize.
- The request input requires image that has been converted into base64.
- The response will return in JSON format </br>
```status:boolean``` -> return status whether the request is successfull. </br>
```Main_Image:base64``` -> return the detection result that has been rendered with bbox in bitmap(bmp) format. </br>
```Cropped_Image:Array of image``` -> return for each bboxed images. </br>
```Predicted:utf8-string``` -> return the predicted labels for each bbox images. </br>
