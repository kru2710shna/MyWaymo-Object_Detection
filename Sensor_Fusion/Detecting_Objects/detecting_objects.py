# ---------------------------------------------------------------------
# Exercises from lesson 2 (object detection)
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.  
#
# Purpose of this file : Examples
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

from PIL import Image
import io
import sys
import os
import cv2
import open3d as o3d
import math
import numpy as np
import zlib

## Add current working directory to path
sys.path.append(os.getcwd())

## Waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
import misc.objdet_tools as tools


# Example C2-4-3 : Display detected objects on top of BEV map
def render_obj_over_bev(detections, lidar_bev_labels, configs, vis=False):

    # project detected objects into bird's eye view
    tools.project_detections_into_bev(lidar_bev_labels, detections, configs, [0,0,255])

    # display bev map
    if vis==True:
        lidar_bev_labels = cv2.rotate(lidar_bev_labels, cv2.ROTATE_180)   
        cv2.imshow("BEV map", lidar_bev_labels)
        cv2.waitKey(0) 



# Example C2-4-3 : Display label bounding boxes on top of bev map
def render_bb_over_bev(bev_map, labels, configs, vis=False):

    # convert BEV map from tensor to numpy array
    bev_map_cpy = (bev_map.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    bev_map_cpy = cv2.resize(bev_map_cpy, (configs.bev_width, configs.bev_height))

    # convert bounding box format format and project into bev
    label_objects = tools.convert_labels_into_objects(labels, configs)
    tools.project_detections_into_bev(bev_map_cpy, label_objects, configs, [0,255,0])
    
    # display bev map
    if vis==True:
        bev_map_cpy = cv2.rotate(bev_map_cpy, cv2.ROTATE_180)   
        cv2.imshow("BEV map", bev_map_cpy)
        cv2.waitKey(0)          

    return bev_map_cpy 

    

# Example C2-4-2 : count total no. of vehicles and vehicles that are difficult to track
def count_vehicles(frame):

    # initialze static counter variables
    if not hasattr(count_vehicles, "cnt_vehicles"):
        count_vehicles.cnt_vehicles = 0
        count_vehicles.cnt_difficult_vehicles = 0

    # loop over all labels
    for label in frame.laser_labels:

        if label.type == label_pb2.Label.Type.TYPE_VEHICLE:
            count_vehicles.cnt_vehicles += 1
            if label.detection_difficulty_level > 0:
                count_vehicles.cnt_difficult_vehicles += 1

    print("no. of labelled vehicles = " + str(count_vehicles.cnt_vehicles) + ", no. of vehicles difficult to detect = " + str(count_vehicles.cnt_difficult_vehicles))


# Example C2-3-3 : Minimum and maximum intensity
def min_max_intensity(lidar_pcl):

    # retrieve min. and max. intensity value from point cloud
    min_int = np.amin(lidar_pcl[:,3])
    max_int = np.amax(lidar_pcl[:,3])

    print("min. intensity = " + str(min_int) + ", max. intensity = " + str(max_int))


# Example C2-3-1 : Crop point cloud
def crop_pcl(lidar_pcl, configs, vis=True):

    # remove points outside of detection cube defined in 'configs.lim_*'
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask]

    # visualize point-cloud
    if vis:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar_pcl)
        o3d.visualization.draw_geometries([pcd])

    return lidar_pcl


# Exercise C2-4-6 : Plotting the precision-recall curve
def plot_precision_recall(): 

    # Please note: this function assumes that you have pre-computed the precions/recall value pairs from the test sequence
    #              by subsequently setting the variable configs.conf_thresh to the values 0.1 ... 0.9 and noted down the results.
    
    # Please create a 2d scatter plot of all precision/recall pairs 
    import matplotlib.pyplot as plt
    P = [0.97, 0.94, 0.93, 0.92, 0.915, 0.91, 0.89, 0.87, 0.82]
    R = [0.738, 0.738, 0.743, 0.746, 0.746, 0.747, 0.748, 0.752, 0.754]
    plt.scatter(R, P)   
    plt.show()

# Exercise C2-3-4 : Compute precision and recall
def compute_precision_recall(det_performance_all, conf_thresh=0.5):

    if len(det_performance_all)==0 :
        print("no detections for conf_thresh = " + str(conf_thresh))
        return
    
    # extract the total number of positives, true positives, false negatives and false positives
    # format of det_performance_all is [ious, center_devs, pos_negs]
    pos_negs = []
    for item in det_performance_all:
        pos_negs.append(item[2])
    pos_negs_arr = np.asarray(pos_negs)        

    positives = sum(pos_negs_arr[:,0])
    true_positives = sum(pos_negs_arr[:,1])
    false_negatives = sum(pos_negs_arr[:,2])
    false_positives = sum(pos_negs_arr[:,3])
    print("TP = " + str(true_positives) + ", FP = " + str(false_positives) + ", FN = " + str(false_negatives))
    
    # compute precision
    precision = true_positives / (true_positives + false_positives) # When an object is detected, what are the chances of it being real?   
    
    # compute recall 
    recall = true_positives / (true_positives + false_negatives) # What are the chances of a real object being detected?

    print("precision = " + str(precision) + ", recall = " + str(recall) + ", conf_thres = " + str(conf_thresh) + "\n")    
    


# Exercise C2-3-2 : Transform metric point coordinates to BEV space
def pcl_to_bev(lidar_pcl, configs, vis=True):

    # compute bev-map discretization by dividing x-range by the bev-image height
    bev_discret = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height

    # create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates    
    lidar_pcl_cpy = np.copy(lidar_pcl)
    lidar_pcl_cpy[:, 0] = np.int_(np.floor(lidar_pcl_cpy[:, 0] / bev_discret))

    # transform all metrix y-coordinates as well but center the foward-facing x-axis on the middle of the image
    lidar_pcl_cpy[:, 1] = np.int_(np.floor(lidar_pcl_cpy[:, 1] / bev_discret) + (configs.bev_width + 1) / 2)

    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl_cpy[:, 2] = lidar_pcl_cpy[:, 2] - configs.lim_z[0]  

    # re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then by decreasing height
    idx_height = np.lexsort((-lidar_pcl_cpy[:, 2], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))
    lidar_pcl_hei = lidar_pcl_cpy[idx_height]

    # extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    _, idx_height_unique = np.unique(lidar_pcl_hei[:, 0:2], axis=0, return_index=True)
    lidar_pcl_hei = lidar_pcl_hei[idx_height_unique]

    # assign the height value of each unique entry in lidar_top_pcl to the height map and 
    # make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    height_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    height_map[np.int_(lidar_pcl_hei[:, 0]), np.int_(lidar_pcl_hei[:, 1])] = lidar_pcl_hei[:, 2] / float(np.abs(configs.lim_z[1] - configs.lim_z[0]))
    
    # sort points such that in case of identical BEV grid coordinates, the points in each grid cell are arranged based on their intensity
    lidar_pcl_cpy[lidar_pcl_cpy[:,3]>1.0,3] = 1.0
    idx_intensity = np.lexsort((-lidar_pcl_cpy[:, 3], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))
    lidar_pcl_cpy = lidar_pcl_cpy[idx_intensity]

    # only keep one point per grid cell
    _, indices = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True)
    lidar_pcl_int = lidar_pcl_cpy[indices]

    # create the intensity map
    intensity_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    intensity_map[np.int_(lidar_pcl_int[:, 0]), np.int_(lidar_pcl_int[:, 1])] = lidar_pcl_int[:, 3] / (np.amax(lidar_pcl_int[:, 3])-np.amin(lidar_pcl_int[:, 3]))

    # visualize intensity map
    if vis:
        img_intensity = intensity_map * 256
        img_intensity = img_intensity.astype(np.uint8)
        while (1):
            cv2.imshow('img_intensity', img_intensity)
            if cv2.waitKey(10) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

