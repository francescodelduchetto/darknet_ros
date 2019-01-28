#!/usr/bin/env python
import cv2
import rospy
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseArray
from darknet_ros_msgs.msg import BoundingBoxes

class PeopleDetectionsIn3DPose():

    def __init__(self, depth_image_topic="/head_xtion/depth/image_rect", bboxes_topic="/darknet_ros/bounding_boxes", camera_info_topic="/head_xtion/depth/camera_info", poses_topic="/darknet_ros/bounding_box_centres"):

        rospy.Subscriber(depth_image_topic, Image, self.depth_image_cb)
        rospy.Subscriber(bboxes_topic, BoundingBoxes, self.bounding_boxes_cb)
        rospy.Subscriber(camera_info_topic, CameraInfo, self.K_cb)

        poseArray_pub = rospy.Publisher(poses_topic, PoseArray, queue_size=1)

        self.cv_bridge = CvBridge()
        self.last_image = None
        self.K = None
        print "initialized"

    def bounding_boxes_cb(self, msg):
        print "bboxes cb"
        if self.last_image is None:
            rospy.logwarn("Bounding boxes received, but no rgb image yet!")
            return
        if self.K is None:
            rospy.logwarn("Bounding boxes received, but no K matrix yet!")
            return

        # cropped_imgs = []
        # centers_2d = []
        # median_distances = []
        pose_array = PoseArray()
        bbs = msg.bounding_boxes
        for bb in bbs:
            if bb.Class != "person":
                continue

            # get depth bboxes
            crop = self.get_depth_bbox(bb.xmin, bb.ymin, bb.xmax, bb.ymax)
            if crop:
                # cropped_imgs.append(crop)

                # get bbox center
                center_x = (bb.xmax - bb.xmin) / 2 + bb.xmin
                center_y = (bb.ymax - bb.ymin) / 2 + bb.ymin
                # centers_2d.append([
                #         center_x,
                #         center_y
                #     ])

                # compute median distance
                md_dist = self.get_median_distance(crop)
                # median_distances.append(md_dist)

                # convert to 3D poses
                pose = Pose()
                # pose.position.x = detected_bounding_boxes(i)(5)*((mid_point_x-K(2,0))/K(0,0));
                # pose.position.y = detected_bounding_boxes(i)(5)*((mid_point_y-K(2,1))/K(1,1));
                # pose.position.z = detected_bounding_boxes(i)(5);
                pose.position.x = md_dist * ((center_x - self.K[2]) / self.K[0])
                pose.position.y = md_dist * ((center_y - self.K[5]) / self.K[4])
                pose.position.z = md_dist

                pose_array.poses.append(pose)

        if len(pose_array.poses) > 0:
            self.publish_poses(pose_array)



    def depth_image_cb(self, msg):
        # print "depth cb"
        self.last_image = msg

    def K_cb(self, msg):
        # print "K cb"
        # save the instrinsic row-major matrix
        self.K = msg.K

    def get_depth_bbox(self, xmin, ymin, xmax, ymax):
        if self.last_image is None:
            return 0

        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(self.last_image)
        except CvBridgeError as e:
            rospy.logerr("CvBridgeErr: " + e)
            return 0

        cropped_img = cv_image[ymin:ymax, xmin:xmax]

        # show image for debug
        cv2.imshow("cropped", cropped_img)
        cv2.waitKey(0)

        return cropped_img

    def get_median_distance(self, img):
        # mask zero values
        masked = np.ma.masked_where(img == 0, img)

        median = np.ma.median(masked).filled(0)

        return median
    #
    # def publish_poses(pose_arr):
    #     self.poseArray_pub.publish(pose_arr)


if __name__ == '__main__':
    rospy.init_node("darknet_3D_pose_node")

    pdi3p = PeopleDetectionsIn3DPose()

    rospy.spin()
