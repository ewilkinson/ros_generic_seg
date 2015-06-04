#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
import cv2

from skimage import morphology
from skimage.filters import threshold_otsu

from sklearn import mixture
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN
from sklearn.preprocessing import StandardScaler

import time


class GenericSegmenter:
    def __init__(self, cluster_type="dbscan", use_gray=True, depth_max_threshold=4000, show_segmentation=False, show_cluster=False, show_mask=False, merge_boxes=False):
        self.cluster_type = cluster_type
        self.use_gray = use_gray

        self.show_segmentation = show_segmentation
        self.show_cluster = show_cluster
        self.show_mask = show_mask
        self.merge_boxes = merge_boxes

        if show_segmentation:
            cv2.namedWindow("Image window", 1)
        if show_mask:
            cv2.namedWindow("Mask window", 1)
        if show_cluster:
            cv2.namedWindow("Cluster window", 1)


        self.gmm = mixture.GMM(n_components=1, covariance_type='full')

        self.bridge = CvBridge()

        self.frame_subsample = 6
        self.count = 0
        self.depth_count = 0

        self.depth_max_threshold = depth_max_threshold
        self.box_merge_threshold = 0.8

        self.depth_mask = None
        self.depth_img = None
        self.rgb_imge = None
        self.gray_image = None
        self.current_depth_thresh = 0

        self.DBSCAN = "dbscan"
        self.MEANSHIFT = "meanshift"

    def rand_color(self):
        return int(np.random.random() * 255)

    def _cluster(self, X):
        if self.cluster_type is self.MEANSHIFT:

            bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=100)
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            ms.fit(X)
            n_clusters = np.unique(ms.labels_).size
            labels = ms.labels_

        elif self.cluster_type is self.DBSCAN:

            X_prime = np.copy(X)
            X_prime = StandardScaler().fit_transform(X_prime)
            db = DBSCAN(eps=0.3, min_samples=10).fit(X_prime)
            n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
            labels = db.labels_
        else:
            raise RuntimeError("Cluster type was not recognized.")

        unique_labels = set(labels)

        return labels, unique_labels, n_clusters

    def _gen_features(self, mask, gray_image):
        x_size, y_size = mask.shape[0], mask.shape[1]
        gX, gY = np.meshgrid(range(mask.shape[0]), range(mask.shape[1]))

        if self.use_gray:
            X = np.zeros((mask.size, 5))
            X[:, 0] = gX.transpose().ravel() * 1.0 / x_size
            X[:, 1] = gY.transpose().ravel() * 1.0 / y_size
            X[:, 2] = mask.ravel()
            X[:, 3] = self.depth_img.ravel() * 1.0 / self.current_depth_thresh
            X[:, 4] = gray_image.ravel() * 1.0 / gray_image.max()
        else:
            X = np.zeros((mask.size, 4))
            X[:, 0] = gX.transpose().ravel() * 1.0 / x_size
            X[:, 1] = gY.transpose().ravel() * 1.0 / y_size
            X[:, 2] = mask.ravel()
            X[:, 3] = self.depth_img.ravel() * 1.0 / self.current_depth_thresh

        X = X[X[:, 2] == 1]
        num_samples = X.shape[0]
        step_size = int(num_samples / 3000)

        if step_size == 0:
            step_size = 1

        X = X[0::step_size, :]

        return X, x_size, y_size

    # works with GMM method
    def _merge_boxes_gmm(self, boxes):
        """
        Merges boxes using the amount of overlap area and a threshold parameters to decide if they should be merged.
        Repeatedly passes over the boxes to ensure every combination is considered.

        :param boxes:
        :return:
        """
        while True:

            new_boxes = []
            has_merged = False

            while len(boxes):
                xa1, ya1, xa2, ya2, mean_depth_a = boxes.pop()
                SA = abs((xa1 - xa2) * (ya1 - ya2))
                this_has_merged = False

                for j in range(len(boxes) - 1, -1, -1):

                    xb1, yb1, xb2, yb2, mean_depth_b = boxes[j]
                    SB = abs((xb1 - xb2) * (yb1 - yb2))
                    SI = max(0, min(xa2, xb2) - max(xa1, xb1)) * max(0, min(ya2, yb2) - max(ya1, yb1))
                    ratio = max(SI * 1.0 / SA, SI * 1.0 / SB)

                    if ratio > self.box_merge_threshold:
                        has_merged = True
                        this_has_merged = True
                        boxes.remove(boxes[j])
                        new_x1, new_x2 = min(xa1, xb1), max(xa2, xb2)
                        new_y1, new_y2 = max(ya1, yb1), min(ya2, yb2)
                        new_mean_depth = SA * mean_depth_a + SB * mean_depth_b / (1.0 * SA + SB)
                        new_boxes.append([new_x1, new_y1, new_x2, new_y2, new_mean_depth])
                        break

                if not this_has_merged:
                    new_boxes.append([xa1, ya1, xa2, ya2, mean_depth_a])

            # break out of infinite loop
            if not has_merged:
                return new_boxes
            else:
                boxes = new_boxes

    def depth_callback(self, data):
        self.depth_count += 1
        if self.depth_count % self.frame_subsample == 0:
            return 0

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
        except CvBridgeError, e:
            print e

        depth_threshold = threshold_otsu(cv_image)

        if depth_threshold > self.depth_max_threshold:
            depth_threshold = self.depth_max_threshold

        cv_image[cv_image == 0] = depth_threshold
        self.depth_mask = np.squeeze(cv_image < depth_threshold)
        self.depth_img = cv_image
        self.current_depth_thresh = depth_threshold

    def callback(self, data):
        self.count += 1
        if self.count % self.frame_subsample == 0:
            return 0

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError, e:
            print e

        start_time = time.time()

        if self.depth_mask is None:
            return

        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        self.rgb_imge = np.copy(cv_image)
        self.gray_image = gray_image

        # masked_gray = gray_image*self.depth_mask
        # mask1 = gray_image < threshold_otsu(masked_gray)
        # mask2 = gray_image > threshold_otsu(masked_gray)
        # mask = mask1 if np.sum(mask1) < np.sum(mask2) else mask2

        mask1 = gray_image < threshold_otsu(gray_image)
        mask2 = gray_image > threshold_otsu(gray_image)
        mask = mask1 if np.sum(mask1) < np.sum(mask2) else mask2

        mask = morphology.remove_small_objects(mask * self.depth_mask, 50)
        mask = np.asarray(mask, dtype=np.uint8)

        proc_time = time.time()
        # Try to localize an object

        X, x_size, y_size = self._gen_features(mask, gray_image)

        boxes = []
        if X.shape[0] > 100:

            labels, unique_labels, n_clusters = self._cluster(X)


            # CLUSTER IMAGE
            if self.show_cluster:
                cluster_image = np.asarray(mask, dtype=np.uint8) * 255
                cluster_image = np.repeat(cluster_image, 3, axis=1).reshape(mask.shape + (3, ))

            for k in unique_labels:

                # this is the sklean label for noise
                if k == -1:
                    continue

                class_member_mask = (labels == k)

                class_feats = X[class_member_mask, :]

                # probably a noisey cluster detection
                if class_feats.shape[0] < 20:
                    continue


                self.gmm.fit(class_feats)
                covars = np.sqrt(np.asarray(self.gmm._get_covars()))

                alpha = 2.2
                x1 = int((self.gmm.means_[0, 0] - alpha * covars[0, 0, 0]) * x_size)
                x2 = int((self.gmm.means_[0, 0] + alpha * covars[0, 0, 0]) * x_size)
                y1 = int((self.gmm.means_[0, 1] - alpha * covars[0, 1, 1]) * y_size)
                y2 = int((self.gmm.means_[0, 1] + alpha * covars[0, 1, 1]) * y_size)
                # d1 = int((self.gmm.means_[0, 3] - alpha * covars[0, 3, 3]) * self.current_depth_thresh)
                # d2 = int((self.gmm.means_[0, 3] + alpha * covars[0, 3, 3]) * self.current_depth_thresh)

                mean_depth = self.gmm.means_[0, 3] * self.current_depth_thresh

                boxes.append([x1, y1, x2, y2, mean_depth])

                if self.show_cluster:
                    exp=5
                    for i in range(class_feats.shape[0]):
                        x, y = int(class_feats[i,0]*x_size), int(class_feats[i,1]*y_size)
                        c = self.gmm.means_[0, 3] * 255
                        cluster_image[max(x- exp,0):min(x+exp, cluster_image.shape[0]),max(y- exp,0):min(y+exp,cluster_image.shape[1]), 0] = c
                        cluster_image[max(x- exp,0):min(x+exp, cluster_image.shape[0]),max(y- exp,0):min(y+exp,cluster_image.shape[1]), 1] = c
                        cluster_image[max(x- exp,0):min(x+exp, cluster_image.shape[0]),max(y- exp,0):min(y+exp,cluster_image.shape[1]), 2] = c

            if self.merge_boxes:
                boxes = self._merge_boxes_gmm(boxes)

            if self.show_segmentation:
                for x1, y1, x2, y2, mean_depth in boxes:
                    cv2.rectangle(cv_image, (y1, x1), (y2, x2), (125,0,0), 2)

            self.boxes = boxes

        end_time = time.time()
        # print 'Pre Proc Time : ', proc_time - start_time
        # print 'Total Time : ', end_time - start_time

        if self.show_mask:
            mask *= 255
            cv2.imshow("Mask window", mask)
            cv2.waitKey(1)

        if self.show_segmentation:
            cv2.imshow("Image window", cv_image)
            cv2.waitKey(1)

        if self.show_cluster:
            cv2.imshow("Cluster window", cluster_image)
            cv2.waitKey(1)


    def listen(self):
        rospy.init_node('generic_seg', anonymous=True)

        rospy.Subscriber("/asus/rgb/image_raw", Image, self.callback)
        rospy.Subscriber("/asus/depth/image_raw", Image, self.depth_callback)


if __name__ == '__main__':
    segmenter = GenericSegmenter(cluster_type="dbscan",
                            use_gray=False,
                            depth_max_threshold=4000,
                            show_segmentation=True,
                            show_cluster=True,
                            show_mask=False,
                            merge_boxes=True)

    try:
        segmenter.listen()

        while True:
            time.sleep(5)

    except KeyboardInterrupt:
        print "Shutting down"

    cv2.destroyAllWindows()

