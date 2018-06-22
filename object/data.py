from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import json
from skimage import io

def aug_norm_transforms():
    """
    Define data augmentation and normalization transforms
    :return:
    """
    # Data augmentation and normalization for training
    # Just normalization for validation
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


class SceneMap(dict):
    def __init__(self, factory):
        super()
        self.factory = factory

    def __missing__(self, key):
        self[key] = self.factory(key)
        return self[key]


class ImageObject:
    def __init__(self, category_id, category_name, bounding_box):
        self.category_id = category_id
        self.category_name = category_name
        self.bounding_box = bounding_box


class Scene:
    def __init__(self, id_, file_name):
        self.id = id_
        self.file_name = file_name
        self.objects = []


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
        scenes = SceneMap(lambda id_: Scene(id_, filenames[id_]))

        # In the json metadata, all annoted objects are stored in a flat list, so we need to resolve
        # to which scene they belong and group them as such.
        for image_object in metadata['annotations']:
            if image_object['ignore']:
                continue
            # The image id associates a scene with the image object
            image_id = image_object['image_id']
            category_id = image_object['category_id']
            scenes[image_id].objects.append(
                ImageObject(category_id, categories[category_id], image_object['bbox'])
            )
        return scene_ids, scenes

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, idx):
        image_id = self.scene_ids[idx]
        image_item = self.scenes[image_id]
        img_name = os.path.join(self.image_root_dir, image_item.file_name)
        # TODO: This might be improved by not using sklearn, but a more efficient method
        image = io.imread(img_name)
        dataset_item = {'image': image, 'scene': image_item}

        # We need to co-transform the bounding boxes if the image gets distorted by the transform
        if self.transform:
            dataset_item = self.transform(dataset_item)
        return dataset_item

