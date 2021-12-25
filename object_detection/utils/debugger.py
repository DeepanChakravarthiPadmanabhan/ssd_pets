import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

import argparse
import numpy as np
from paz.models import SSD300
from paz.abstract import Processor, SequentialProcessor
from paz import processors as pr
from object_detection.trainer.pipelines import AugmentDetection
from object_detection.dataloader.pets import PetsDetection
from object_detection.dataloader.pets import class_names_list


class ShowBoxes(Processor):
    def __init__(self, class_names, prior_boxes,
                 variances=[0.1, 0.1, 0.2, 0.2]):
        super(ShowBoxes, self).__init__()
        self.deprocess_boxes = SequentialProcessor([
            pr.DecodeBoxes(prior_boxes, variances),
            pr.ToBoxes2D(class_names, True),
            pr.FilterClassBoxes2D(class_names[1:])])
        self.denormalize_boxes2D = pr.DenormalizeBoxes2D()
        self.draw_boxes2D = pr.DrawBoxes2D(class_names)
        self.show_image = pr.ShowImage()
        self.resize_image = pr.ResizeImage((600, 600))

    def call(self, image, boxes):
        image = self.resize_image(image)
        boxes2D = self.deprocess_boxes(boxes)
        boxes2D = self.denormalize_boxes2D(image, boxes2D)
        image = self.draw_boxes2D(image, boxes2D)
        image = (image + pr.BGR_IMAGENET_MEAN).astype(np.uint8)
        image = image[..., ::-1]
        self.show_image(image)
        return image, boxes2D


description = 'Training script for single-shot object detection models'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-dsp', '--dataset_path',
                    default='/media/deepan/externaldrive1/datasets_project_repos/pets_data',
                    type=str, help='Marine debris dataset path')
args = parser.parse_args()

Dataset = PetsDetection(args.dataset_path, 'trainval', class_names_list)
train_data = Dataset.load_data()
class_names = Dataset.class_names

model = SSD300(head_weights=None, base_weights=None, num_classes=3)
prior_boxes = model.prior_boxes

testor_encoder = AugmentDetection(prior_boxes, num_classes=3)
testor_decoder = ShowBoxes(class_names, prior_boxes)
sample_ids = [100, 1232, 12, 124, 443]
for sample_arg in sample_ids:
    sample = train_data[sample_arg]
    wrapped_outputs = testor_encoder(sample)
    image = wrapped_outputs['inputs']['image']
    boxes = wrapped_outputs['labels']['boxes']
    image, boxes = testor_decoder(image, boxes)
