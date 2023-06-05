import json
import os
import argparse
import shutil
import numpy as np


def coco_to_yolov5(coco_file, output_dir, type="train"):
    with open(coco_file, "r") as f:
        coco_data = json.load(f)

    images = coco_data["images"]
    annotations = coco_data["annotations"]

    os.makedirs(output_dir, exist_ok=True)
    image_output_dir = os.path.join(output_dir, "images", type)
    labels_output_dir = os.path.join(output_dir, "labels", type)

    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)

    for image in images:
        image_id = image["id"]
        file_name = image["file_name"]
        width = image["width"]
        height = image["height"]

        shutil.move(
            os.path.join(os.path.dirname(coco_file), "img", file_name), image_output_dir
        )

        image_annotations = [
            anno for anno in annotations if anno["image_id"] == image_id
        ]

        output_file = os.path.join(
            labels_output_dir, os.path.splitext(file_name)[0] + ".txt"
        )
        with open(output_file, "w") as f:
            for annotation in image_annotations:
                category_name = 0

                # bbox = annotation["bbox"]
                # bbox_x = bbox[0]
                # bbox_y = bbox[1]
                # bbox_width = bbox[2]
                # bbox_height = bbox[3]

                # x_center = bbox_x + bbox_width / 2
                # y_center = bbox_y + bbox_height / 2

                # x_center /= width
                # y_center /= height
                # bbox_width /= width
                # bbox_height /= height

                # f.write(
                #     f"{category_name} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"
                # )

                s = [
                    j for i in annotation["segmentation"] for j in i
                ]  # all segments concatenated
                s = (
                    (np.array(s).reshape(-1, 2) / np.array([width, height]))
                    .reshape(-1)
                    .tolist()
                )
                s = [category_name] + s
                f.write(("%g " * len(s)).rstrip() % tuple(s) + "\n")

    print(f"Conversion complete. YOLOv5 inputs saved to {output_dir}.")


def generate_image_list(coco_file, output_file):
    with open(coco_file, "r") as f:
        coco_data = json.load(f)

    images = coco_data["images"]

    image_list = [image["file_name"] for image in images]

    with open(output_file, "w") as f:
        f.write("\n".join(image_list))

    print(f"Image list generated. Saved to {output_file}.")


def __main__():
    """_summary_ = Main function to convert COCO annotations to YOLOv5 format."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_file_path", type=str, required=True)
    # parser.add_argument("--output_dir", type=str, required=True)
    # parser.add_argument("--type", type=str, default="train")
    args = parser.parse_args()

    coco_to_yolov5(
        os.path.join(args.coco_file_path, "coco_train.json"),
        args.coco_file_path,
        "train",
    )
    generate_image_list(
        os.path.join(args.coco_file_path, "coco_train.json"),
        os.path.join(args.coco_file_path, "train.txt"),
    )
    coco_to_yolov5(
        os.path.join(args.coco_file_path, "coco_val.json"),
        args.coco_file_path,
        "val",
    )
    generate_image_list(
        os.path.join(args.coco_file_path, "coco_val.json"),
        os.path.join(args.coco_file_path, "val.txt"),
    )


__main__()
