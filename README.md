# Japanese character recognition Flask API with YOLOv4
This repositories is adapting YOLOv4 to Flask API with Darkeras [here](https://github.com/tranleanh/darkeras-yolov4).
The client to this API is targeted to be Java Android using retrofit.

## Installation

<b>This API runs on Python 3.8</b>

<b>Weight placement :</b>
- Place your trained weight and obj.names to weight folder

<b>Darkeras instalation and various changes are necessarily required inorder to make the API runs :</b>
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
