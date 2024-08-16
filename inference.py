import sys
import os
import torch
import random
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


class Detectron2Setup:
    def __init__(self):
        self.cfg = get_cfg()

    def setup_environment(self):
        setup_logger()
        torch_version = ".".join(torch.__version__.split(".")[:2])
        cuda_version = torch.__version__.split("+")[-1]
        print(f"torch: {torch_version}; cuda: {cuda_version}")
        print(f"detectron2: {detectron2.__version__}")

    def configure_model(self, output_dir, num_classes=1, model_config="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"):
        self.cfg.OUTPUT_DIR = output_dir
        self.cfg.merge_from_file(model_zoo.get_config_file(model_config))
        self.cfg.DATASETS.TRAIN = ("my_dataset_train",)
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.SOLVER.BASE_LR = 0.00025
        self.cfg.SOLVER.MAX_ITER = 10000
        self.cfg.SOLVER.STEPS = []
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        self.cfg.MODEL.DEVICE = "cpu"

    def train_model(self, resume=False):
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=resume)
        trainer.train()

    def load_trained_model(self, model_weights_filename):
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, model_weights_filename)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
        return DefaultPredictor(self.cfg)


class DatasetManager:
    def __init__(self, dataset_name, json_path, image_dir):
        self.dataset_name = dataset_name
        self.json_path = json_path
        self.image_dir = image_dir

    def register_dataset(self):
        register_coco_instances(self.dataset_name, {}, self.json_path, self.image_dir)
        metadata = MetadataCatalog.get(self.dataset_name)
        dataset_dicts = DatasetCatalog.get(self.dataset_name)
        return metadata, dataset_dicts


class InferenceVisualizer:
    def __init__(self, predictor, metadata, output_dir):
        self.predictor = predictor
        self.metadata = metadata
        self.output_dir = output_dir

    def visualize_predictions(self, dataset_dicts):
        for d in random.sample(dataset_dicts, 1):
            im = cv2.imread(d["file_name"])

            outputs = self.predictor(im)

            original_count = len(d['annotations'])

            v = Visualizer(im[:, :, ::-1], metadata=self.metadata, scale=1, instance_mode=ColorMode.IMAGE_BW)
            v = v.draw_dataset_dict(d)
            original_with_boxes = v.get_image()[:, :, ::-1]

            prediction_count = len(outputs['instances'])

            v = Visualizer(im[:, :, ::-1], metadata=self.metadata, scale=1, instance_mode=ColorMode.IMAGE_BW)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            self._save_and_show_image(original_with_boxes, "Original Image with Original Bounding Boxes", "original_image_with_boxes.png")
            print(f"Number of LED components in original image: {original_count}")

            self._save_and_show_image(out.get_image()[:, :, ::-1], "Image with Predictions", "image_with_predictions.png")
            print(f"Number of LED components in image with predictions: {prediction_count}")

    def _save_and_show_image(self, image, title, filename):
        plt.figure(figsize=(image.shape[1] / 100, image.shape[0] / 100))
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.show()


class Evaluator:
    def __init__(self, cfg, dataset_name):
        self.cfg = cfg
        self.dataset_name = dataset_name

    def evaluate(self, predictor):
        evaluator = COCOEvaluator(self.dataset_name, output_dir="./output")
        test_loader = build_detection_test_loader(self.cfg, self.dataset_name)
        return inference_on_dataset(predictor.model, test_loader, evaluator)


def main():
    # Setup and configuration
    detectron2_setup = Detectron2Setup()
    detectron2_setup.setup_environment()
    detectron2_setup.configure_model(
        output_dir="/Users/asad/Documents/pcbdata/COCO_Data_Single_Component/LED/saved/Models",
        num_classes=1
    )
    detectron2_setup.train_model(resume=False)

    # Load trained model
    predictor = detectron2_setup.load_trained_model("model_10k_iter.pth")

    # Register datasets
    train_manager = DatasetManager(
        dataset_name="my_dataset_train",
        json_path="/Users/asad/Documents/pcbdata/COCO_Data_Single_Component/LED/train/images/Coco_LED_only_train.json",
        image_dir="/Users/asad/Documents/pcbdata/COCO_Data_Single_Component/LED/train/images"
    )
    val_manager = DatasetManager(
        dataset_name="my_dataset_val",
        json_path="/Users/asad/Documents/pcbdata/COCO_Data_Single_Component/LED/validation/images/Coco_LED_only_val.json",
        image_dir="/Users/asad/Documents/pcbdata/COCO_Data_Single_Component/LED/validation/images"
    )
    test_manager = DatasetManager(
        dataset_name="my_dataset_test",
        json_path="/Users/asad/Documents/pcbdata/COCO_Data_Single_Component/LED/test/images/Coco_LED_only_test.json",
        image_dir="/Users/asad/Documents/pcbdata/COCO_Data_Single_Component/LED/test/images"
    )

    _, test_dataset_dicts = test_manager.register_dataset()

    # Visualize predictions
    visualizer = InferenceVisualizer(predictor, MetadataCatalog.get("my_dataset_test"), "/Users/asad/Documents/pcbdata/COCO_Data_Single_Component/LED")
    visualizer.visualize_predictions(test_dataset_dicts)

    # Evaluate model
    evaluator = Evaluator(detectron2_setup.cfg, "my_dataset_test")
    evaluation_results = evaluator.evaluate(predictor)
    print(evaluation_results)


if __name__ == "__main__":
    main()
