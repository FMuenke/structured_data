import os
import numpy as np
import cv2
from structured_data.augmentations import Augmentations
from structured_data.temporal_spatial_nodes import TemporalSpatialNode, GroupOfTemporalSpatialNodes


class UnstructuredImage(TemporalSpatialNode):
    def __init__(self, index, time_stamp, coordinates, path_to_image, source=None, augment=None):
        self.path_to_image = path_to_image
        self.source = source
        self.augment = augment
        self.representation = None
        assert os.path.isfile(self.path_to_image), "Image is not present at {}".format(self.path_to_image)
        super().__init__(index, time_stamp, coordinates, representation=None, sample=None)

    def set_representation(self, repr):
        self.representation = repr

    def get_repr(self):
        return self.representation

    def load_data(self):
        if self.augment is not None:
            img = cv2.imread(self.path_to_image)
            return self.augment.apply(img)
        return cv2.imread(self.path_to_image)
    
    def copy(self, id_ext):
        return UnstructuredImage(
            index="{}-{}".format(self.index, id_ext),
            time_stamp=self.time_stamp,
            coordinates=self.get_coords(),
            path_to_image=self.path_to_image,
            source=self.source,
            augment=self.augment
        )

    def copy_clean(self, id_ext):
        return UnstructuredImage(
            index="{}-{}".format(self.index, id_ext),
            time_stamp=self.time_stamp,
            coordinates=self.get_coords(),
            path_to_image=self.path_to_image,
            source=self.source,
            augment=None
        )


class GroupOfUnstructruedImages(GroupOfTemporalSpatialNodes):
    def __init__(self, list_of_unstructured_images):
        super().__init__(list_of_unstructured_images)

    def set_augment(self, augment):
        for u_img in self:
            u_img.augment = augment

    def synth_clusters(self, augment, n_multply):
        self.set_augment(augment)
        list_of_goi = []
        for u_img in self:
            goi = GroupOfUnstructruedImages([u_img.copy(i) for i in range(n_multply)])
            list_of_goi.append(goi)
        return list_of_goi
    
    def multiply(self, augment, n_multply):
        self.set_augment(augment)
        goi = GroupOfUnstructruedImages([])
        for u_img in self:
            for i in range(n_multply):
                goi.add(u_img.copy(i))
        return goi
    
    def sort_by_source(self):
        list_of_ids = [u_img.source for u_img in self]
        separated_grps = {source: GroupOfUnstructruedImages([]) for source in np.unique(list_of_ids)}
        for u_img in self:
            separated_grps[u_img.source].add(u_img)
        return separated_grps