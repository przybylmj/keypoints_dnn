import blenderproc as bproc
import numpy as np
import json

def sample_pose(obj: bproc.types.MeshObject):
    obj.set_location(np.random.uniform([-1, -1, 1.5], [1, 1, 2.5]))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())


def main():
    bproc.init()

    objs_path = ["oats.blend","tea.blend","crisps.blend","soup.blend","scissors.blend","pan.blend"]

    for cat_id, path in enumerate(objs_path):

        obj = bproc.loader.load_blend(path)[0]
        if cat_id == 0 or cat_id == 3:
            obj.set_rotation_euler([0, np.pi/2, 0])
        if cat_id == 1:
            obj.set_rotation_euler([np.pi/2,0,0])
        if cat_id == 5:
            obj.set_rotation_euler([-np.pi/2,0,0])
        
        light = bproc.types.Light()
        light.set_type("SUN")
        light.set_location([0, 0, 0])
        light.set_rotation_euler([-0.063, 0.6177, -0.1985])
        light.set_energy(1)
        light.set_color([1, 0.978, 0.407])
        # light.set_location([8,8,8])
        # light.set_energy(300)

        obj.set_cp("category_id",cat_id + 1)
        # r_step=1
        # r_max=3
        # r_min=1
        # Radius=[r for r in range(r_min,r_max+1,int((r_max-r_min)/(r_step)))]
        Radius = [1.5,2.5]
        print(f"radius  {Radius}")
        phi_step=6
        # Phi=[p for p in range(0,360,int(360/(phi_step-1)))]
        Phi = [0, 72, 144, 216, 288]
        print(f"Phi  {Phi}")
        # theta_step=5
        # Theta=[t for t in range(0,90,int(90/(theta_step-1)))]
        Theta = [0, 22, 44, 66, 88]
        print(f"Theta   {Theta}")

        for r in Radius:
            for phi in Phi:
                for theta in Theta:
                    print(f"R phi theta:    {r} {phi} {theta}")
                    bproc.utility.reset_keyframes()
                    phi_rad=phi*np.pi/180
                    theta_rad=theta*np.pi/180
                    x=r*np.cos(theta_rad)*np.cos(phi_rad)
                    y=r*np.cos(theta_rad)*np.sin(phi_rad)
                    z=r*np.sin(theta_rad)
                    bproc.camera.set_resolution(800,800)                    
                    cam_pose = bproc.math.build_transformation_mat([x,y,z], [np.pi/2-theta_rad, 0, np.pi/2+phi_rad])
                    bproc.camera.add_camera_pose(cam_pose)
                    data = bproc.renderer.render()
                    seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])
                    bproc.writer.write_coco_annotations("ds_raw_simple/",
                                                        instance_segmaps=seg_data["instance_segmaps"],
                                                        instance_attribute_maps=seg_data["instance_attribute_maps"],
                                                        colors=data["colors"],
                                                        color_file_format="JPEG",
                                                        mask_encoding_format="polygon")

        obj.delete()    

    with open("ds_raw_simple/coco_annotations.json") as coco_file:
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
    with open("ds_raw_simple/coco_annotations.json","w") as coco_file:
        json.dump(coco,coco_file)


if __name__ == '__main__':
    main()