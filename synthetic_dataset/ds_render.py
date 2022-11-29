import blenderproc as bproc
import numpy as np
import bpy
import bpy_extras
from mathutils import Vector
import math
import json

def sample_pose(obj: bproc.types.MeshObject):
    obj.set_location(np.random.uniform([-1, -1, 1.5], [1, 1, 2.5]))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())


def main():
    bproc.init()
    ground = bproc.loader.load_obj("board.obj")[0]
    oats = bproc.loader.load_blend("oats.blend")[0]
    tea  = bproc.loader.load_blend("tea.blend")[0]
    crisps = bproc.loader.load_blend("crisps.blend")[0]
    soup = bproc.loader.load_blend("soup.blend")[0]
    scissors = bproc.loader.load_blend("scissors.blend")[0]
    pan = bproc.loader.load_blend("pan.blend")[0]
    
    light = bproc.types.Light()
    light.set_type("SUN")
    light.set_location([0, 0, 0])
    light.set_rotation_euler([-0.063, 0.6177, -0.1985])
    light.set_energy(1)
    light.set_color([1, 0.978, 0.407])
    # light.set_location([8,8,8])
    # light.set_energy(300)

    ground.enable_rigidbody(active=False,collision_shape="MESH")

    oats.enable_rigidbody(active=True)
    tea.enable_rigidbody(active=True)
    crisps.enable_rigidbody(active=True)
    soup.enable_rigidbody(active=True)
    scissors.enable_rigidbody(active=True)
    pan.enable_rigidbody(active=True)
    
    oats.set_cp("category_id",1)
    tea.set_cp("category_id",2)
    crisps.set_cp("category_id",3)
    soup.set_cp("category_id",4)
    scissors.set_cp("category_id",5)
    pan.set_cp("category_id",6)
    
    objects = [oats,tea,crisps,soup,scissors,pan]

    cam_poses = [[[1.2, -1.2, 2.5], [0.6, 0, (np.pi/4)]],
                [[1.2, 1.2, 2.5], [0.6, 0, (3*np.pi/4)]],
                [[-1.2, 1.2, 2.5], [0.6, 0, (5*np.pi/4)]],
                [[-1.2, -1.2, 2.5], [0.6, 0, (7*np.pi/4)]]]   

    previous_ann_id = 1

    for i in  range(60):
        bproc.camera.set_resolution(800,800)
        bproc.object.sample_poses(objects,sample_pose_func=sample_pose)
        bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=4, max_simulation_time=20, check_object_interval=1)

        for pose in cam_poses:
            bproc.utility.reset_keyframes()
            cam_pose = bproc.math.build_transformation_mat(pose[0], pose[1])
            bproc.camera.add_camera_pose(cam_pose)
            data = bproc.renderer.render()
            seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])
            bproc.writer.write_coco_annotations("ds_raw/",
                                                instance_segmaps=seg_data["instance_segmaps"],
                                                instance_attribute_maps=seg_data["instance_attribute_maps"],
                                                colors=data["colors"],
                                                color_file_format="JPEG",
                                                mask_encoding_format="polygon")


    with open("ds_raw/coco_annotations.json") as coco_file:
        coco = json.load(coco_file)
    for img in coco["images"]:
        img["id"] = img["id"] + 1
    for ann in coco["annotations"]:
        ann["image_id"] = ann["image_id"] + 1
        ann["segmentation"] = []
        if(False):   ### SET BBOX MARGINS ###
            margin=int(0.1*(ann['bbox'][2] + ann['bbox'][3])/2)
            ann['bbox'][0] = ann['bbox'][0] - margin 
            ann['bbox'][1] = ann['bbox'][1] - margin
            ann['bbox'][2] = ann['bbox'][2] + 2*margin
            ann['bbox'][3] = ann['bbox'][3] + 2*margin
    with open("ds_raw/coco_annotations.json","w") as coco_file:
        json.dump(coco,coco_file)


if __name__ == '__main__':
    main()