import argparse
from PIL import Image
import yaml

import torch
from torch.utils.data import DataLoader

from datasets import get_dataset
from models import load_model_from_path
from utils import coerce_to_path_and_check_exist, coerce_to_path_and_create_dir
from utils.constant import MODEL_FILE
from utils.image import LabeledArray2Image, resize
from utils.logger import print_info
from utils.metrics import RunningMetrics
from utils.path import MODELS_PATH


import numpy as np

class Tester:
    """Pipeline to test a given trained NN model on the test split of a specified dataset."""

    def __init__(self, output_dir, model_path, dataset_name, dataset_kwargs=None, save_annotations=True):
        print_info("Tester initialized for model {} and dataset {}".format(model_path, dataset_name))
        

        # Output directory
        self.output_dir = coerce_to_path_and_create_dir(output_dir)
        self.save_annotations = save_annotations
        print_info("Output dir is {}".format(self.output_dir))

        # Dataset
        self.dataset_kwargs = dataset_kwargs or {}
        #self.dataset_kwargs['restricted_labels_1']
        self.dataset = get_dataset(dataset_name)("test", self.dataset_kwargs['restricted_labels_2'], **self.dataset_kwargs)
        print_info("Dataset {} loaded with kwargs {}: {} samples"
                   .format(dataset_name, self.dataset_kwargs, len(self.dataset)))
                   
        self.batch_size = cfg["training"]["batch_size"]
        self.n_workers = cfg["training"]["n_workers"]
        #self.batch_size = 1
        #self.n_workers = 4
        self.test_loader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.n_workers)
        

        # Model
        torch.backends.cudnn.benchmark = False  # XXX: at inference, input images are usually not fixed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model_from_path(model_path, device=self.device)
        self.model.eval()
        print_info("Model {} created and checkpoint state loaded".format(self.model.name))

        # Metrics
        if self.dataset.label_files is not None:
            self.metrics = RunningMetrics(self.dataset.restricted_labels, self.dataset.metric_labels)
            self.current_metrics = RunningMetrics(self.dataset.restricted_labels)
            print_info("Labels found, metrics instantiated")
        else:
            self.metrics = None
            print_info("No labels found, performance metrics won't be computed")

        # Outputs
        # saving probability maps takes a lot of space, remove comment if needed
        # self.prob_dir = coerce_to_path_and_create_dir(self.output_dir / "prob_map")
        self.prob_maps, self.seg_maps = [], []
        if self.save_annotations:
            self.seg_dir = coerce_to_path_and_create_dir(self.output_dir / "seg_map")
            self.blend_dir = coerce_to_path_and_create_dir(self.output_dir / "blend")
            
        self.tables_metrics = []
        
        
    def run(self):
        #for image, label in self.test_loader:
        for idx, data in enumerate(self.test_loader):
            
            (image,label) = data
            
            print(self.dataset.input_files[idx])
        
            self.single_run(image, label)
        print_info("Probabilities and segmentation maps computed")

        if self.metrics is not None:
            self.save_metrics()

        metrics = self.metrics.get()
        self.print_and_log_info("Test metrics: " + ", ".join(["{} = {:.4f}".format(k, v) for k, v in metrics.items()]))
        
        print(np.nanmean(self.tables_metrics))
        print(np.nanstd(self.tables_metrics))
        
        print_info("Run is over")

    @torch.no_grad()
    def single_run(self, image, label=None):
        image = image.to(self.device)
        label = label.to(self.device)
        ####
        prob = self.model(image)[1]
        tables_metrics = []
        
        pred = prob.max(1)[1].cpu().numpy()
        self.prob_maps.append(prob.cpu().numpy())
        self.seg_maps.append(pred)

        
        if label is not None:
            if image.size() == label.size():
                gt = label.data.max(1)[1].cpu().numpy()
            else:
                gt = label.cpu().numpy()

            self.metrics.update(gt, pred)
            self.current_metrics.update(gt, pred)
            self.print_and_log_info("Test metrics: " + ", ".join(["{} = {:.4f}".format(k, v) for k, v in self.current_metrics.get().items()]))
            self.tables_metrics.append(self.current_metrics.get()['iou_class_9'])
            self.current_metrics.reset()

    def save_metrics(self):
        with open(self.output_dir / "test_metrics.tsv", mode="w") as f:
            f.write("\t".join(self.metrics.names) + "\n")
            f.write("\t".join(map("{:.4f}".format, self.metrics.get().values())) + "\n")

        print_info("Metrics saved")

    def save_prob_and_seg_maps(self):
        for k in range(len(self.dataset)):
            name = self.dataset.input_files[k].stem
            # saving probability maps takes a lot of space, remove comment if needed
            # np.save(self.prob_dir / "{}.npy".format(name), self.prob_maps[k])
            pred = self.seg_maps[k]
            pred_img = LabeledArray2Image.convert(pred, label_color_mapping=self.dataset.label_idx_color_mapping)
            pred_img.save(self.seg_dir / "{}.png".format(name))

            img = resize(Image.open(self.dataset.input_files[k]), pred_img.size, keep_aspect_ratio=False)
            blend_img = Image.blend(img, pred_img, alpha=0.4)
            blend_img.convert("RGB").save(self.blend_dir / "{}.jpg".format(name))

        print_info("Probabilities and segmentation maps saved")

        
    def print_and_log_info(self, string):
        print_info(string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline to test a NN model on the test split of a dataset")
    parser.add_argument("-t", "--tag", nargs="?", type=str, help="Model tag to test", required=True)
    parser.add_argument("-d", "--dataset", nargs="?", type=str, default="syndoc", help="Name of the dataset to test")
    args = parser.parse_args()

    run_dir = coerce_to_path_and_check_exist(MODELS_PATH / args.tag)
    output_dir = run_dir / "test_{}".format(args.dataset)
    config_path = list(run_dir.glob("*.yml"))[0]
    with open(config_path) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    dataset_kwargs = cfg["dataset"]
    dataset_kwargs.pop("name_1")
    dataset_kwargs.pop("name_2")
    dataset_kwargs.pop("name_3")
    dataset_kwargs.pop("name_4")
    

    tester = Tester(output_dir, run_dir / MODEL_FILE, args.dataset, dataset_kwargs)
    tester.run()
