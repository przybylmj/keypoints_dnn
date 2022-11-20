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

def mean(items):
    sum_x = 0
    sum_y = 0
    sum_z = 0 
    for item in items:
        sum_x = sum_x + item[0]
        sum_y = sum_y + item[1]
        sum_z = sum_z + item[2]
    return Vector(((sum_x/len(items)),(sum_y/len(items)),(sum_z/len(items))))

def getBBoxPointsLocal3d(obj):
    dim = bpy.data.objects[obj.get_name()].dimensions
    pom = dim/2
    verts =[Vector((pom[0],pom[1],pom[2])),
            Vector((pom[0],pom[1],-pom[2])),
            Vector((pom[0],-pom[1],pom[2])),
            Vector((pom[0],-pom[1],-pom[2])),
            Vector((-pom[0],pom[1],pom[2])),
            Vector((-pom[0],pom[1],-pom[2])),
            Vector((-pom[0],-pom[1],pom[2])),
            Vector((-pom[0],-pom[1],-pom[2]))]
    return verts

def getBBoxPointWorld3d(obj,verts_local):
    rot = bpy.data.objects[obj.get_name()].rotation_euler
    alfa, beta, gamma = rot
    transl = bpy.data.objects[obj.get_name()].location
    T = np.array([[1,0,0,transl[0]],[0,1,0,transl[1]],[0,0,1,transl[2]],[0,0,0,1]])
    Rx = np.array([[1,0,0,0],[0,math.cos(alfa),-math.sin(alfa),0],[0,math.sin(alfa),math.cos(alfa),0],[0,0,0,1]])
    Ry = np.array([[math.cos(beta),0,math.sin(beta),0],[0,1,0,0],[-math.sin(beta),0,math.cos(beta),0],[0,0,0,1]])
    Rz = np.array([[math.cos(gamma),-math.sin(gamma),0,0],[math.sin(gamma),math.cos(gamma),0,0],[0,0,1,0],[0,0,0,1]])
    M = T @ Rx @ Ry @ Rz
    print(f"mat: {M}")
    verts_world = []
    for vert in verts_local:
        vert_local = [vert[0],vert[1],vert[2],1]
        vert_world = M @ vert_local
        verts_world.append(Vector((vert_world[0],vert_world[1],vert_world[2])))
    return verts_world    

def getKeypoints3d(verts):
    walls = [[1,2,3,4],[5,6,7,8],[1,2,5,6],[3,4,7,8],[1,3,5,7],[2,4,6,8]]
    wall_central_pts = []
    for wall in walls:
        wall_pts = []
        for id in wall:
            wall_pts.append(verts[id - 1])
        wall_central_pts.append(mean(wall_pts))
    grasp_pt = max(wall_central_pts,key= lambda x: x[2])
    print(f"GRASP_PT:   {grasp_pt}")
    grasp_pt_id = wall_central_pts.index(grasp_pt)
    print(f"GRASP_PT_ID:    {grasp_pt_id}")
    grasp_wall = walls[grasp_pt_id]
    print(f"GRASP_WALL:   {grasp_wall}")
    keypoints3d = [grasp_pt,verts[grasp_wall[0]-1],verts[grasp_wall[1]-1],verts[grasp_wall[2]-1],verts[grasp_wall[3]-1]]
    print("####GRASP_WALL_PTS#####")
    for pt in keypoints3d:
        print(pt)
    return keypoints3d

def world2camera(keypoints3d):
    render_scale = bpy.context.scene.render.resolution_percentage / 100
    render_size = (int(bpy.context.scene.render.resolution_x * render_scale),int(bpy.context.scene.render.resolution_y * render_scale))
    keypoints2d = []
    for pt3d in keypoints3d:
        pt2d=bpy_extras.object_utils.world_to_camera_view(bpy.context.scene,bpy.data.objects['Camera'],pt3d)
        keypoints2d.append(round(pt2d.x * render_size[0]))
        keypoints2d.append(bpy.context.scene.render.resolution_y - round(pt2d.y * render_size[1]))
        keypoints2d.append(2)
    return keypoints2d

def main():
    bproc.init()

    objs_path = ["oats.blend","tea_box_obj.blend","crisps_can_obj.blend","soup_can_obj.blend","scissors.blend","pan.blend"]

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
        objects = [obj]
        previous_ann_id = 1
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
        theta_step=5
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

                    for obj in objects:
                        print(f'###### {obj.get_name()} ######')
                        obj_local_3d_verts =  getBBoxPointsLocal3d(obj)
                        obj_world_3d_verts =  getBBoxPointWorld3d(obj,obj_local_3d_verts)
                        for vert in obj_world_3d_verts:
                            print(vert)
                        keypoints3d = getKeypoints3d(obj_world_3d_verts)
                        keypoints2d = world2camera(keypoints3d)
                        print(f"KEYPOINTS 2D:   {keypoints2d}")
                        with open("ds_raw_simple/coco_annotations.json") as coco_file:
                            coco = json.load(coco_file)
                        current_ann_id = len(coco["annotations"])
                        for ann_id in range(previous_ann_id,current_ann_id + 1): 
                            coco["annotations"][ann_id-1]["segmentation"] = []
                            if coco["annotations"][ann_id-1]["category_id"] == obj.get_cp("category_id"):
                                coco["annotations"][ann_id-1]["num_keypoints"] = 1 #5
                                coco["annotations"][ann_id-1]["keypoints"] = [] #keypoints2d
                        with open("ds_raw_simple/coco_annotations.json","w") as coco_file:
                            json.dump(coco,coco_file)
                    previous_ann_id = current_ann_id + 1
        obj.delete()    

    with open("ds_raw_simple/coco_annotations.json") as coco_file:
        coco = json.load(coco_file)
    for img in coco["images"]:
        img["id"] = img["id"] + 1
    for ann in coco["annotations"]:
        ann["image_id"] = ann["image_id"] + 1
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