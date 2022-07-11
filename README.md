# `tableTranscriber`: an automatic pipeline for astronomical tables transcription

**tableTranscriber** is based on [docExtractor](https://github.com/monniert/docExtractor) (work of Tom Monnier) and a CRNN [HTR method](https://github.com/vloison/Handwritten_Text_Recognition) (work of Virginie Loison and Xiwei Hu) to automatically recognize astronomical tables structure, and transcribe their content. It can be easily re-trained on new synthetic datasets, or fine-tuned on real world data.

For a technical explanation on how tableTranscriber works, please read our [work report](work_report_tableTranscriber.pdf).

![illustration](illustration_pipeline.jpg)


## Installation

### Prerequisites

You must have Anaconda installed on your computer.

### 1. Create conda environment

```bash
conda env create -f environment.yml
conda activate tableTranscriber
```

### 2. Download resources and models

To download our models and datasets, please enter the following command:

```bash
./download.sh
```

It will download: 
- **Structure recognition**:
    - our trained model for astronomical table structure recognition (located: `tableExtractor/models/four_branches_tables`)
    - docExtractor's default trained model (created by Tom Monnier, located: `tableExtractor/models/default`)
    - synthetic resources needed to generate SynDoc (collected by Tom Monnier, located: `tableExtractor/synthetic_resource`)
	- a synthetic dataset with 10K images and their corresponding ground truths for the 4 branches of our network (located: `tableExtractor/datasets/synthetic_tables`) ; the synthetic images contain tables and diagrams inspired by astronomical manuscripts
    - DishasTables dataset (located: `tableExtractor/datasets/DishasTables`)
- **HTR**:
    - our trained model for HTR on astronomical tables (located: `HTR/trained_networks/medieval_numbers.pth`)
    - our HTR training dataset (composed of cells of medieval astronomical tables, located: `HTR/datasets/tables_cells`)


## Dataset management

### Generation of new synthetic tables

```
python tableExtractor/src/syndoc_generator.py -d dataset_name -n nb_train --merged_labels --table_multi_labels
```
Main arguments:
- `-d, --dataset_name`: name of the folder containing the synthetic dataset
- `-n, --nb_train`: number of training samples to generate
- `-m, --merged_labels`: whether to merge all graphical and textual labels into unique `illustration` and `text` labels (the `table`, `line` and `column` elements are not merged) 
- `-t, --table_multi_labels`: whether or not to generate 4 distinct datasets, to deal with multi label classification (in order to train a network with 4 branches, necessary for the recognition of tables structures)

The training datasets have the following structure:
- they are composed of three distinct `train`, `val` and `test` folders
- in these folders, each `name.jpg` input image is linked to a corresponding `name_seg.png` ground truth label image

The synthetic dataset layouts, and element compositions can be easily changed by modifying:
- the variables defined in `tableExtractor/src/synthetic/document.py` (and especially the variables `LAYOUT_RANGE` and `ELEMENT_FREQ`)
- the classes defined in `tableExtractor/src/synthetic/element.py` (and especially the classes `TableElement_loose`, `TableElement_compact`, `DiagramElement`)

**N.B.**: All the labels, and corresponding colors are defined in `tableExtractor/src/utils/constant.py`. 

### Creation of HTR dataset

The HTR dataset structure must follow the [IAM](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) dataset structure. Our training dataset, `HTR/datasets/tables_cells`, composed of ~1700 transcribed cells from 6 manuscripts, was annotated by hand.

The wanted dataset structure is the following one:

```
Data folder / 
    dataset/
        cells.txt
        cells/
        split/
            trainset.txt
            testset.txt
            validationset1.txt
            validationset2.txt
```

## Training

### Segmentation training:
In order to train a new segmentation neural network: 

```bash
CUDA_VISIBLE_DEVICES=gpu_id python tableExtractor/src/trainer.py --config file --tag tag
```

Main arguments:
- `-c, --config`: name of the training configuration file (e.g. `syndoc_4_branches.yml`)
- `-t, --tag`: name given to the newly trained neural network

The important file is the config file in `.yml`, which especially gives the localization for the ground truth dataset for each branch of the network. Moreover, the data augmentation, models and learning parameters are defined in this config file.

In this config file are also defined the restricted labels on which are trained each one of the branches of the neural network (labels defined in `tableExtractor/src/utils/constant.py`).

Some important labels, for table analysis are:
- `TABLE_LABEL` = 9
- `LINE_LABEL` = 13
- `COLUMN_LABEL` = 14

### HTR training

To train the HTR network, the command is the following one:

```bash
CUDA_VISIBLE_DEVICES=gpu_id python HTR/train.py --tr_data_path data_dir --save_model_path path --save True --epoch number_epoch --data_aug True --enhance_contrast True
```

The training parameters are close to the one described in the `README` of the [original HTR repo](https://github.com/vloison/Handwritten_Text_Recognition) we used for this project.

Main arguments:
- `--tr_data_path`: path of the training dataset
- `--save_model_path`: path of the saved neural network, if `--save` is set to True
- `--epoch`: number of epochs
- `--data_aug`: whether to apply random affine transformations to the input data
- `--enhance_contrast`: whether to enhance the contrast of the images, as a pre-processing step

### Fine-tune network on custom segmentation datasets

Concerning fine-tuning, please see the corresponding [docExtractor](https://github.com/monniert/docExtractor) `README` explanations, especially concerning the use of [VGG Image Anotator](http://www.robots.ox.ac.uk/~vgg/software/via/) and `tableExtractor/src/via_converter.py` tool to create the corresponding annotated images training dataset. 

To fine tune a neural network, the `pretrained` variable of the training `.yml` config file must correspond to a pre-trained network `.pkl` (for instance `default`, or `four_branches_tables`). An example of fine-tuning config file can be found at `tableExtractor/configs/syndoc_4_branches_fine_tuning.yml` – in this case, the fine-tuning is made on the columns and rows separators labels (on the branches adapted to make `tableExtractor/src/transcriber.py` works correctly), as explained in the comments at the beginning of the file.

In our case, since we are working with 4 branches neural networks, you need to freeze the losses you are not interested in during the fine-tuning.
In `tableExtractor/src/trainer.py`: please modify the `loss` variables definitions, in the functions `single_train_batch_run` and `run_val`, to take into account only the branches of interest in your fine-tuning on 4 branches.

## How to use: `tableTranscriber`

To transcribe astronomical tables, please use the following command:

```bash
CUDA_VISIBLE_DEVICES=gpu_id python tableExtractor/src/transcriber.py --input_dir inp --output_dir out --tag tag --save_annot
```

Main arguments:
- `-i, --input_dir`: directory with images of tables to transcribe
- `-o, --output_dir`: directory where transcribed tables will be saved
- `-t, --tag`: model used for table segmentation (e.g. `four_branches_tables`)
- `-s, --save_annot`: whether to save annotated segmentation images
- `-sb, --straight_bbox`: whether to use straight bounding boxes instead of rotated ones to fit connected components

**N.B.**: Each input image should contain 0 or 1 table (no more than 1).

In outputs, `transcriber.py` will create folders of cropped indexed cells (depending on the rows and columns separators, in the `cell_images` folder), automatically transcribed (transcription can be found in `cell_images/predictions.txt`). Fully transcribed tables are also created in `.html` and `.xml` file formats.
