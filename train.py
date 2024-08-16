import sys
import os
import distutils.core
import torch
import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import cv2
import random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
import yaml


class Detectron2Setup:
    def __init__(self, repo_url, setup_file):
        self.repo_url = repo_url
        self.setup_file = setup_file

    def clone_and_install(self):
        os.system(f'git clone {self.repo_url}')
        dist = distutils.core.run_setup(self.setup_file)
        os.system(f'python -m pip install {" ".join([f"\'{x}\'" for x in dist.install_requires])}')
        sys.path.insert(0, os.path.abspath('./detectron2'))
        setup_logger()

    def check_versions(self):
        torch_version = ".".join(torch.__version__.split(".")[:2])
        cuda_version = torch.__version__.split("+")[-1]
        print("torch: ", torch_version, "; cuda: ", cuda_version)
        print("detectron2:", detectron2.__version__)


class DatasetRegistration:
    def __init__(self, dataset_name, json_path, image_dir):
        self.dataset_name = dataset_name
        self.json_path = json_path
        self.image_dir = image_dir

    def register_dataset(self):
        register_coco_instances(self.dataset_name, {}, self.json_path, self.image_dir)
        metadata = MetadataCatalog.get(self.dataset_name)
        dataset_dicts = DatasetCatalog.get(self.dataset_name)
        return metadata, dataset_dicts


class Detectron2Trainer:
    def __init__(self, output_dir, train_dataset, num_classes, config_file="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"):
        self.cfg = get_cfg()
        self.cfg.OUTPUT_DIR = output_dir
        self.cfg.merge_from_file(model_zoo.get_config_file(config_file))
        self.cfg.DATASETS.TRAIN = (train_dataset,)
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.SOLVER.BASE_LR = 0.00025
        self.cfg.SOLVER.MAX_ITER = 10000
        self.cfg.SOLVER.STEPS = []
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    def train(self, resume=False):
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=resume)
        trainer.train()

    def save_config_and_rename_model(self, config_filename, model_final_filename):
        config_yaml_path = os.path.join(self.cfg.OUTPUT_DIR, config_filename)
        with open(config_yaml_path, 'w') as file:
            yaml.dump(self.cfg, file)

        model_final_path = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        os.rename(model_final_path, os.path.join(self.cfg.OUTPUT_DIR, model_final_filename))


def main():
    # Setup Detectron2
    detectron2_setup = Detectron2Setup(
        repo_url="https://github.com/facebookresearch/detectron2",
        setup_file="./detectron2/setup.py"
    )
    detectron2_setup.clone_and_install()
    detectron2_setup.check_versions()

    # Register datasets
    train_registration = DatasetRegistration(
        dataset_name="my_dataset_train",
        json_path="/kaggle/input/coco-ic/train/images/Coco_IC_only_train.json",
        image_dir="/kaggle/input/coco-ic/train/images"
    )
    val_registration = DatasetRegistration(
        dataset_name="my_dataset_val",
        json_path="/kaggle/input/coco-ic/validation/images/Coco_IC_only_val.json",
        image_dir="/kaggle/input/coco-ic/validation/images"
    )

    train_metadata, train_dataset_dicts = train_registration.register_dataset()
    val_metadata, val_dataset_dicts = val_registration.register_dataset()

    # Train the model
    trainer = Detectron2Trainer(
        output_dir="/kaggle/working/models/Detectron2_Models",
        train_dataset="my_dataset_train",
        num_classes=1
    )
    trainer.train(resume=False)
    trainer.save_config_and_rename_model(
        config_filename="config-10k_iter.yaml",
        model_final_filename="model_10k_iter.pth"
    )


if __name__ == "__main__":
    main()
