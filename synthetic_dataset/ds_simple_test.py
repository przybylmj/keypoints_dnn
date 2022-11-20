import json
import cv2 as cv
import os

with open("ds_raw_simple/coco_annotations.json") as coco_file:
    coco = json.load(coco_file)

for img in coco["images"]:
    path = os.path.join("ds_raw_simple",img["file_name"])
    print(path)
    img_id = img["id"]
    image = cv.imread(path)
    for ann in coco["annotations"]:
        if ann["image_id"] == img_id:
            print(ann["id"])
            box = ann["bbox"]
            kp_set = ann["keypoints"]
            print(box)
            print(kp_set)
            cat = ann["category_id"]
            image = cv.rectangle(image,(box[0],box[1]),(box[0] + box[2],box[1] + box[3]),(255,0,255),3)
            image= cv.circle(image,(kp_set[0],kp_set[1]),4,(255,0,255),5) #grasping point
    cv.imshow("output",image)
    cv.imwrite(f"ds_gt_simple/img_gt_{img_id}.jpg",image)
