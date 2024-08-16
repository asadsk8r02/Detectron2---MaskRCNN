import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import json
from PIL import Image


class FileCleaner:
    def __init__(self, annotation_dir):
        self.annotation_dir = annotation_dir

    def remove_empty_csv_files(self):
        annotation_files = os.listdir(self.annotation_dir)

        for annotation_file in annotation_files:
            if annotation_file.endswith(".csv"):
                df = pd.read_csv(os.path.join(self.annotation_dir, annotation_file))
                if df.empty:
                    os.remove(os.path.join(self.annotation_dir, annotation_file))
                    print(f"Deleted {annotation_file} as it is empty.")


class ImageAnnotationSynchronizer:
    def __init__(self, image_dir, annotation_dir):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir

    def sync_images_with_annotations(self):
        image_files = os.listdir(self.image_dir)

        for image_file in image_files:
            image_name, _ = os.path.splitext(image_file)
            annotation_file = image_name + "_smd.csv"

            if not os.path.exists(os.path.join(self.annotation_dir, annotation_file)):
                os.remove(os.path.join(self.image_dir, image_file))
                print(f"Deleted {image_file} as corresponding annotation mask file does not exist.")


class DatasetSplitter:
    def __init__(self, image_dir, annotation_dir, train_dir, test_dir, val_dir):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir

    def create_folders_if_not_exist(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            os.makedirs(os.path.join(directory, "images"))
            os.makedirs(os.path.join(directory, "annotations"))

    def split_dataset(self):
        self.create_folders_if_not_exist(self.train_dir)
        self.create_folders_if_not_exist(self.test_dir)
        self.create_folders_if_not_exist(self.val_dir)

        image_files = os.listdir(self.image_dir)
        train_images, test_images = train_test_split(image_files, test_size=0.2, random_state=42)
        train_images, val_images = train_test_split(train_images, test_size=0.2, random_state=42)

        self._move_files(train_images, self.train_dir)
        self._move_files(test_images, self.test_dir)
        self._move_files(val_images, self.val_dir)

    def _move_files(self, images, target_dir):
        for image in images:
            image_path = os.path.join(self.image_dir, image)
            annotation_file = image.split(".")[0] + "_smd.csv"
            annotation_path = os.path.join(self.annotation_dir, annotation_file)
            shutil.move(image_path, os.path.join(target_dir, "images", image))
            shutil.move(annotation_path, os.path.join(target_dir, "annotations", annotation_file))


class CocoConverter:
    def __init__(self, image_dir, csv_dir, output_file_path, object_category_map):
        self.image_dir = image_dir
        self.csv_dir = csv_dir
        self.output_file_path = output_file_path
        self.object_category_map = object_category_map
        self.coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        self.annotation_id = 0

    def convert_to_coco_format(self):
        self._add_categories_to_coco()
        self._process_csv_files()
        self._save_coco_data()

    def _add_categories_to_coco(self):
        for category_name, category_id in self.object_category_map.items():
            self.coco_data["categories"].append({"id": category_id, "name": category_name})

    def _process_csv_files(self):
        for csv_file in os.listdir(self.csv_dir):
            if not csv_file.endswith(".csv"):
                continue

            csv_path = os.path.join(self.csv_dir, csv_file)
            image_name = csv_file.replace("_smd.csv", ".png")
            image_path = os.path.join(self.image_dir, image_name)

            with open(csv_path, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                self._process_csv_rows(reader, image_name, image_path)

    def _process_csv_rows(self, reader, image_name, image_path):
        for row in reader:
            vertices = eval(row["Vertices"])
            designator = row["Designator"]

            if designator not in self.object_category_map:
                continue

            if image_name not in [image["file_name"] for image in self.coco_data["images"]]:
                image = Image.open(image_path)
                image_info = {
                    "id": len(self.coco_data["images"]),
                    "width": image.width,
                    "height": image.height,
                    "file_name": image_name
                }
                self.coco_data["images"].append(image_info)

            self.annotation_id += 1
            category_id = self.object_category_map[designator]
            segmentation = self.convert_segmentation(vertices)
            bbox = self.calculate_bbox(vertices)
            area = self.calculate_area(bbox)
            self.coco_data["annotations"].append({
                "id": self.annotation_id,
                "image_id": len(self.coco_data["images"]) - 1,
                "category_id": category_id,
                "segmentation": segmentation,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            })

    def _save_coco_data(self):
        with open(self.output_file_path, "w") as jsonfile:
            json.dump(self.coco_data, jsonfile)

    @staticmethod
    def convert_segmentation(vertices):
        segmentation = []
        for segment in vertices:
            segment_points = []
            for vertex in segment:
                segment_points.extend(vertex)
            segmentation.append(segment_points)
        return segmentation

    @staticmethod
    def calculate_bbox(vertices):
        x_min = float('inf')
        y_min = float('inf')
        x_max = float('-inf')
        y_max = float('-inf')

        for segment in vertices:
            for point in segment:
                x, y = point
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

        width = x_max - x_min
        height = y_max - y_min
        return [x_min, y_min, width, height]

    @staticmethod
    def calculate_area(bbox):
        width, height = bbox[2], bbox[3]
        return width * height


def main():
    # File cleaner
    annotation_dir = "/Users/asad/Documents/pcbdata/COCO_Data_All_Component/new_merged_removed_final"
    file_cleaner = FileCleaner(annotation_dir)
    file_cleaner.remove_empty_csv_files()

    # Image and annotation synchronizer
    image_dir = "/Users/asad/Documents/pcbdata/COCO_Data_All_Component/pro_pcb_images"
    synchronizer = ImageAnnotationSynchronizer(image_dir, annotation_dir)
    synchronizer.sync_images_with_annotations()

    # Dataset splitter
    train_dir = "/Users/asad/Documents/pcbdata/COCO_Data_All_Component/train"
    test_dir = "/Users/asad/Documents/pcbdata/COCO_Data_All_Component/test"
    val_dir = "/Users/asad/Documents/pcbdata/COCO_Data_All_Component/validation"
    splitter = DatasetSplitter(image_dir, annotation_dir, train_dir, test_dir, val_dir)
    splitter.split_dataset()

    # Convert train set to COCO format
    object_category_map = {
        "Background": 0, "TL": 1, "D": 2, "J": 3, "SC": 4, "SW": 5, "VR": 6, "CO": 7, "IC": 8, "C": 9, "L": 10,
        "TR": 11, "R": 12, "A": 13, "CR": 14, "RN": 15, "TP": 16, "QA": 17, "LED": 18, "K": 19, "TB": 20, "T": 21,
        "H": 22, "F": 23, "PL": 24, "BTN": 25, "JP": 26, "E": 27, "CN": 28, "Z": 29, "BT": 30, "ZD": 31, "FIL": 32,
        "FB": 33, "PT": 34, "FL": 35, "LD": 36, "RV": 37, "AR": 38
    }
    
    # Train
    coco_converter_train = CocoConverter(
        image_dir=os.path.join(train_dir, "images"),
        csv_dir=os.path.join(train_dir, "annotations"),
        output_file_path="/Users/asad/Documents/pcbdata/COCO_Data_All_Component/train/Coco_All_Categories_train.json",
        object_category_map=object_category_map
    )
    coco_converter_train.convert_to_coco_format()

    # Validation
    coco_converter_val = CocoConverter(
        image_dir=os.path.join(val_dir, "images"),
        csv_dir=os.path.join(val_dir, "annotations"),
        output_file_path="/Users/asad/Documents/pcbdata/COCO_Data_All_Component/validation/Coco_All_Categories_val.json",
        object_category_map=object_category_map
    )
    coco_converter_val.convert_to_coco_format()

    # Test
    coco_converter_test = CocoConverter(
        image_dir=os.path.join(test_dir, "images"),
        csv_dir=os.path.join(test_dir, "annotations"),
        output_file_path="/Users/asad/Documents/pcbdata/COCO_Data_All_Component/test/Coco_All_Categories_test.json",
        object_category_map=object_category_map
    )
    coco_converter_test.convert_to_coco_format()


if __name__ == "__main__":
    main()
