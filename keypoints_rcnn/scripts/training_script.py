
import os
import numpy as np
import torch
import torchvision
from PIL import Image
import pycocotools
from torchvision import transforms
from torchvision.models.detection.rpn import AnchorGenerator
from utils import collate_fn
import transforms, utils, engine, train
from engine import train_one_epoch, evaluate
import os
import argparse
from azureml.core import Run
run = Run.get_context()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(f"Device type:    {device}")

parser = argparse.ArgumentParser()
parser.add_argument("--data-folder",type=str,dest="data_folder",help="path to folder with data",default="")
parser.add_argument("--num-epochs",type=int,dest="num_epochs",help="number of epochs to train model",default=30)

args = parser.parse_args()
data_folder_path = args.data_folder 
num_epochs = args.num_epochs


class ObjectsDataset(torch.utils.data.Dataset):
    def __init__(self,ds_root,ann_file,transform=None):
        self.ds_root=ds_root
        self.ann_file=ann_file
        self.raw_coco_ds=torchvision.datasets.CocoDetection(self.ds_root,self.ann_file)
        self.transform=transform
    
    def __getitem__(self,idx):
        image=self.raw_coco_ds[idx][0]
        boxes,labels,image_id,area,iscrowd,keypoints=[],[],[],[],[],[]
        image_id.append(self.raw_coco_ds[idx][1][0]['image_id'])
        for item_id, item in enumerate(self.raw_coco_ds[idx][1]):
            bbox_xywh=item['bbox']
            boxes.append([bbox_xywh[0], bbox_xywh[1],bbox_xywh[0] + bbox_xywh[2],bbox_xywh[1] + bbox_xywh[3]])
            labels.append(item['category_id'])
            area.append(item['area'])
            iscrowd.append(item['iscrowd'])
            raw_keypoints=self.raw_coco_ds[idx][1][item_id]['keypoints']
            kps=[]
            for i in range(0,len(raw_keypoints),3):
                kp=[]
                kp.append(raw_keypoints[i])
                kp.append(raw_keypoints[i+1])
                if raw_keypoints[i+2] > 1:
                    kp.append(1)
                else:
                    kp.append(raw_keypoints[i+2])
                kps.append(kp)
            keypoints.append(kps)   
        image=torchvision.transforms.functional.to_tensor(image)
        target={}
        target['boxes'] = torch.as_tensor(boxes,dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels,dtype=torch.int64)
        target['image_id'] = torch.as_tensor(image_id, dtype=torch.int64)
        target['area'] = torch.as_tensor(area,dtype=torch.float32)
        target['iscrowd'] = torch.as_tensor(iscrowd,dtype=torch.int64)
        target['keypoints'] = torch.as_tensor(keypoints,dtype=torch.float32)

        return image, target
        
    def __len__(self):
        return len(self.raw_coco_ds)

def get_keypoints_model():
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model=torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,pretrained_backbone=True,num_keypoints=1,num_classes=7,rpn_anchor_generator=anchor_generator)
    return model

def main():

    dataset_train_root=os.path.join(data_folder_path,"train_images")
    ann_train_file=os.path.join(data_folder_path,"img_train_ann.json")
    dataset_test_root=os.path.join(data_folder_path,"test_images")
    ann_test_file=os.path.join(data_folder_path,"img_test_ann.json")
    dataset_train = ObjectsDataset(dataset_train_root,ann_train_file)
    dataset_test = ObjectsDataset(dataset_test_root,ann_test_file)

    dataset_train_loader=torch.utils.data.DataLoader(dataset_train,batch_size=6,shuffle=True,num_workers=8,collate_fn=collate_fn)
    dataset_test_loader=torch.utils.data.DataLoader(dataset_test,batch_size=2,shuffle=False,num_workers=8,collate_fn=collate_fn)

    model=get_keypoints_model()
    model.to(device)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
    # num_epochs = num_epochs + 1

    for epoch in range(1,num_epochs + 1):
        train_one_epoch(model,optimizer,dataset_train_loader,device,epoch,print_freq=10)
        lr_scheduler.step()
        evaluate(model,dataset_test_loader,device)    
        if epoch % 5 == 0 and epoch >=10:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict()
            },os.path.join('./outputs',f'checkpoint_epoch{epoch}.pth'))

    torch.save(model,'./outputs/kp_model.pth')


if __name__ == "__main__":
    main()
