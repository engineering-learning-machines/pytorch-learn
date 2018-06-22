from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import json
from skimage import io

# def aug_norm_transforms():
#     """
#     Define data augmentation and normalization transforms
#     :return:
#     """
#     # Data augmentation and normalization for training
#     # Just normalization for validation
#     return {
#         'train': transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'valid': transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#     }


class SceneMap(dict):
    def __init__(self, factory):
        super()
        self.factory = factory

    def __missing__(self, key):
        self[key] = self.factory(key)
        return self[key]


class PascalDataset(Dataset):
    """
    Pascal VOC Dataset
    """
    def __init__(self, json_file, image_root_dir, transform=None):
        """

        :param json_file:
        :param image_root_dir:
        :param transform:
        """
        # Try to parse the annotation metadata and convert that information into a list of scenes.
        # We call an item of this data set a scene, because it contains the image, as well as a
        # list of annotated objects with each of its category and a bounding box.
        with open(json_file, 'r') as fp:
            self.scene_ids, self.scenes = self.parse_annotations(json.load(fp))

        self.image_root_dir = image_root_dir
        self.transform = transform

    @staticmethod
    def parse_annotations(metadata):
        # Use these as temporary lookup tables to build the scenes
        categories = {category['id']: category['name'] for category in metadata['categories']}
        filenames = {img['id']: img['file_name'] for img in metadata['images']}

        # This list is used for indexing of the dataset items.
        scene_ids = [img['id'] for img in metadata['images']]
        # This is a collections.defaultdict - like object, but it uses the missing key to
        # automatically add the missing scene in order to be able to append image objects to it.
        scenes = SceneMap(lambda id_: {'id': id_, 'filename': filenames[id_], 'objects': []})

        # In the json metadata, all annoted objects are stored in a flat list, so we need to resolve
        # to which scene they belong and group them as such.
        for scene_object in metadata['annotations']:
            if scene_object['ignore']:
                continue
            # The image id associates a scene with the image object
            image_id = scene_object['image_id']
            category_id = scene_object['category_id']
            # The PyTorch DataLoader doesn't like custom objects, since it tries to batch all of the
            # information in a dataset, so we can't use custom classes for the scene.
            # scenes[image_id].objects.append({
            scenes[image_id]['objects'].append({
                'category_id': category_id,
                'category_name': categories[category_id],
                'bounding_box': scene_object['bbox']
            })
        return scene_ids, scenes

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, idx):
        image_id = self.scene_ids[idx]
        scene = self.scenes[image_id]
        img_name = os.path.join(self.image_root_dir, scene['filename'])
        # TODO: This might be improved by not using sklearn, but a more efficient method
        image = io.imread(img_name)

        # Return this or transform first
        dataset_item = {'image': image, 'scene': scene}
        # We need to co-transform the bounding boxes if the image gets distorted by the transform
        if self.transform:
            dataset_item = self.transform({'image': image, 'scene': scene})
        return dataset_item

    # def __add__(self, other):
    #     super().__add__(other)
