import cv2
import numpy as np
import torch


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, target_size):
        assert isinstance(target_size, (int, tuple))
        self.target_size = target_size

    @staticmethod
    def transform_scene_object(matrix, scene_object):
        """

        :param matrix:
        :param scene_object:
        :return:
        """
        scene_object['bounding_box'] = np.dot(matrix, scene_object['bounding_box'])
        return scene_object

    def __call__(self, item):
        # We need to transform both the image and the bounding boxes
        image, scene = item['image'], item['scene']
        target_size = [0, 0]
        # If the output size is an integer, then the resulting image should have equal width and height
        if isinstance(self.target_size, int):
            target_size[0] = self.target_size
            target_size[1] = self.target_size
        else:
            # Somebody might pass floats or whatever
            target_size = [int(sz) for sz in self.target_size]

        # Transform the image first
        resized_image = cv2.resize(image, tuple(target_size), interpolation=cv2.INTER_AREA)

        # Now let's build a transformation matrix for the bounding boxes. The bounding boxes are not vectors,
        # so we are going to build a 4x4 matrix where the diagonals are the scaling factors:

        matrix = np.eye(4, 4)
        height_factor = target_size[0] / float(image.shape[0])
        width_factor = target_size[1] / float(image.shape[1])

        matrix[0, 0] = width_factor
        matrix[1, 1] = height_factor
        matrix[2, 2] = width_factor
        matrix[3, 3] = height_factor

        # Transform all scene objects
        scene['objects'] = [self.transform_scene_object(matrix, so) for so in scene['objects']]
        return {'image': resized_image, 'scene': scene}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, item):
        image, scene = item['image'], item['scene']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'scene': scene}
