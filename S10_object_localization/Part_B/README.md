## COCO dataset info
Relevant information regarding COCO is available in the https://cocodataset.org/ page. <br>
For the analysis we are using the [2017 Train/Val annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) _Note this file is 241MB_
The training images are also available in the same site, and it is 18GB+. <br>
There are also dataset for other tasks and types, like panoptic segmentation available in that page. <br>


## COCO Annotation Data Format
The COCO Annotations file (2017) has the following keys -- `dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])`. <br>
There are several keys within each of those 👆, we are going to look at only a relevant few here. <br>

- #### `images`
the `images` key contains the info related to the image like the file_name, unique id for each image, height and width of the image, and the image url; among other info. Looking at the length of `images` key (ie.. `len(annotation_json['images'])`) we can see there are 118287 unique images in the dataset. <br>
This is the sample of 1 image -- 
```js
{'license': 3,
 'file_name': '000000559665.jpg',
 'coco_url': 'http://images.cocodataset.org/train2017/000000559665.jpg',
 'height': 397,
 'width': 640,
 'date_captured': '2013-11-15 15:25:41',
 'flickr_url': 'http://farm8.staticflickr.com/7322/9296558150_0fe28321a1_z.jpg',
 'id': 559665}
 ```

- #### `annotations`
Looking at the length of `annotations` key (ie.. `len(annotation_json['annotations'])`) we can see there are 860001 unique annotations in the dataset. <br>
This is the sample of 1 annotation (note: the `segementation` key is truncated here for better visiblity.) <br>
```js
{'segmentation': [[247.71,
   252.53,
   353.73]],
 'area': 1545.4213000000007,
 'iscrowd': 0,
 'image_id': 200365,
 'bbox': [234.22, 317.11, 149.39, 38.55],
 'category_id': 58,
 'id': 509}
 ```
 We also see an `id` for each annotation, and we also see there is a `category_id` linking to the class of the image in the annotation. There are few more fields but we are only going to look at `bbox` for now. The `bbox` key contains a list of four floating point values. And they are ordered as follows --`[x,y,width,height]`, where `x,y` is the pixel of the top left point. Here `x-axis` spans from left to right, and `y-axis` is from top to bottom of the image. <br> 
So `0,0` will correspond to the top left corner of the image. `width` and `height` are the width and height of the bounding box in pixels. <br>
Here are some statistics for the bbox -- <br> (We can see that they are scaled from 0 to 640, that is why the values are in float although its representing pixels, which is discrete 🤔) <br>
![image](https://user-images.githubusercontent.com/8600096/126770905-dc4727ff-640f-4464-9a6e-5a4362dd89c8.png)


- #### `categories`
In the following section we go into detail about the `categories` key, which contains the category mapping. But here is an instance of a category --
```js
{'supercategory': 'person', 'id': 1, 'name': 'person'}
```


## COCO Categories
COCO has 91 classes listed in their info page, but there are only data points for 80 categories available in the dataset.  <br>
The classes are of regular everyday objects, animals so on. Broadly the categories belongs to these super categories - vehicle, outdoor, animals, sports, food, furniture, electronics, so on... <br>
Each super category can have several classes, for example super category - "vehicle" has bicycle, car, plane, bus, truck, train, boat. <br>
This variety of classes makes COCO dataset quite userful in pretraining models for many real world applications. <br>
You can find the category mapping [here](https://github.com/askmuhsin/eva_experiments/blob/main/S10_object_localization/Part_B/coco_category_mapping.json) <br>
Here is a breakdown of the classes (from the 2017 Train/Val annotations file which can be downloaded from [here](https://cocodataset.org/#download)) - <br>
![image](https://user-images.githubusercontent.com/8600096/126772783-7d57c431-3d85-48f1-b1d1-cb2b8627bc69.png)
As we can observe `person` is the most common class. (6X more `person` class annotations than `car`, which is the next biggest class) <br> 
![image](https://user-images.githubusercontent.com/8600096/126773185-2c64171a-ca43-4c04-b309-0331d6e28132.png)


## Extract category mapping info from COCO annotations
In order to extract from the website follow these steps --  <br>

```bash
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
```
```python
import json

ins_train = json.load(open('./annotations/instances_train2017.json', 'r'))
categories = ins_train['categories']
category_mapping = {r['id']:r['name'] for r in categories}
with open('./coco_category_mapping.json', 'w') as fw:
    json.dump(category_mapping, fw)
```
_(To clean up)_
```bash
rm ./annotations_trainval2014.zip
rm -rf ./annotations
```
