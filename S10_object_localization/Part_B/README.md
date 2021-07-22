## COCO category mapping
Relevant information regarding COCO is available in the https://cocodataset.org/ page. <br>
The COCO annotation file is also available in the site. One of the info in the file is the category mapping info. <br>
You can find the category mapping [here](https://github.com/askmuhsin/eva_experiments/blob/main/S10_object_localization/Part_B/category_mapping.json) in this repo. <br>
_Note : the category_mapping above is 0 indexed_ <br>
In order to extract from the website follow these steps --  <br>

```bash
! wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
! unzip annotations_trainval2014.zip
```
```python
import json

ins_train = json.load(open('./annotations/instances_train2014.json', 'r'))
categories = ins_train['categories']
category_mapping = {r['id'] - 1:r['name'] for r in categories}
with open('./category_mapping.json', 'w') as fw:
    json.dump(category_mapping, fw)
```
_(To clean up)_
```bash
rm ./annotations_trainval2014.zip
rm -rf ./annotations
```
