import os
import argparse
import json

from labelme import utils
import numpy as np
import glob
import PIL.Image
import time


class labelme2coco(object):
    def __init__(self, labelme_json=[], save_json_path="./coco.json"):
        """
        :param labelme_json: the list of all labelme json file paths
        :param save_json_path: the path to save new json
        """
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.info= []
        self.images = []
        self.licenses = []
        self.categories = []
        self.annotations = []
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.date = time.strftime('%Y/%m/$d')

        self.save_json()

    def data_transfer(self):
        for num, json_file in enumerate(self.labelme_json):
            with open(json_file, "r") as fp:
                data = json.load(fp)
                self.images.append(self.image(data, num))
                for shapes in data["shapes"]:
                    label = shapes["label"].split("_")
                    if label not in self.label:
                        self.label.append(label)
                    points = shapes["points"]
                    self.annotations.append(self.annotation(points, label, num))
                    self.annID += 1

        # Sort all text labels so they are in the same order across data splits.
        self.label.sort()
        for label in self.label:
            self.categories.append(self.category(label))
        for annotation in self.annotations:
            annotation["caption"] = self.getcatid(annotation["caption"])

    def info(self):
        info = {
            "description": "FMIPA UGM Dataset",
            "url": "http://fmipa.ugm.ac.id",
            "version": "1.0",
            "year": 2014,
            "contributor": "FMIPA UGM",
            "date_created": self.date
        }

        return info
    
    def image(self, data, num):
        image = {}
        img = utils.img_b64_to_arr(data["imageData"])
        height, width = img.shape[:2]
        img = None
        image["license"] = 1
        image["file_name"] = data["imagePath"].split("/")[-1]
        image["height"] = height
        image["width"] = width
        image["date_captured"] = "2019-12-16",
        image["id"] = num

        self.height = height
        self.width = width

        return image

    def annotation(self, points, label, num):
        annotation = {}
        contour = np.array(points)
        x = contour[:, 0]
        y = contour[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        annotation["image_id"] = num
        annotation["caption"] = label[0]  # self.getcatid(label)
        annotation["id"] = self.annID
        annotation["segmentation"] = [list(np.asarray(points).flatten())]
        annotation["iscrowd"] = 0
        annotation["area"] = area
        annotation["bbox"] = list(map(float, self.getbbox(points)))
        return annotation

    def category(self, label):
        category = {}
        category["supercategory"] = label[0]
        category["id"] = len(self.categories)
        category["name"] = label[0]
        return category
    
    def licenses(self):
        licenses = {
            {
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"
            },
            {
                "url": "http://creativecommons.org/licenses/by-nc/2.0/",
                "id": 2,
                "name": "Attribution-NonCommercial License"
            },
            {
                "url": "http://creativecommons.org/licenses/by-nc-nd/2.0/",
                "id": 3,
                "name": "Attribution-NonCommercial-NoDerivs License"
            },
            {
                "url": "http://creativecommons.org/licenses/by/2.0/",
                "id": 4,
                "name": "Attribution License"
            },
            {
                "url": "http://creativecommons.org/licenses/by-sa/2.0/",
                "id": 5,
                "name": "Attribution-ShareAlike License"
            },
            {
                "url": "http://creativecommons.org/licenses/by-nd/2.0/",
                "id": 6,
                "name": "Attribution-NoDerivs License"
            },
            {
                "url": "http://flickr.com/commons/usage/",
                "id": 7,
                "name": "No known copyright restrictions"
            },
            {
                "url": "http://www.usa.gov/copyright.shtml",
                "id": 8,
                "name": "United States Government Work"
            },{
                "url": "Unknow",
                "id": 0,
                "name": "Unknow"
            }
        }
        return licenses

    def getcatid(self, label):
        for category in self.categories:
            if label == category["name"]:
                return category["id"]
        print("label: {} not in categories: {}.".format(label, self.categories))
        exit()
        return -1

    def getbbox(self, points):
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):

        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]

        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        return [
            left_top_c,
            left_top_r,
            right_bottom_c - left_top_c,
            right_bottom_r - left_top_r,
        ]

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {}
        data_coco["info"] = self.info
        data_coco["images"] = self.image
        data_coco["licenses"] = self.licenses
        data_coco["categories"] = self.categories
        data_coco["annotations"] = self.annotations
        return data_coco

    def save_json(self):
        print("save coco json")
        self.data_transfer()
        self.data_coco = self.data2coco()

        print(self.save_json_path)
        os.makedirs(
            os.path.dirname(os.path.abspath(self.save_json_path)), exist_ok=True
        )
        json.dump(self.data_coco, open(self.save_json_path, "w"), indent=4)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="labelme annotation to coco data json file."
    )
    parser.add_argument(
        "labelme_images",
        help="Directory to labelme images and annotation json files.",
        type=str,
    )
    parser.add_argument(
        "--output", help="Output json file path.", default="trainval.json"
    )
    args = parser.parse_args()
    labelme_json = glob.glob(os.path.join(args.labelme_images, "*.json"))
    labelme2coco(labelme_json, args.output)
