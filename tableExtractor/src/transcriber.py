import argparse
import cv2
from PIL import Image, ImageDraw

import numpy as np
import torch

import re
from bisect import bisect

import shutil
from pathlib import Path

from models import load_model_from_path
from utils import coerce_to_path_and_check_exist, coerce_to_path_and_create_dir, get_files_from_dir
from utils.constant import BACKGROUND_LABEL, ILLUSTRATION_LABEL, TEXT_LABEL, LABEL_TO_COLOR_MAPPING, MODEL_FILE, TABLE_LABEL, BASELINE_LABEL, LINE_LABEL, COLUMN_LABEL
from utils.image import Pdf2Image, resize
from utils.logger import get_logger, print_info, print_error
from utils.path import MODELS_PATH
import statistics

from numpy import asarray

import os

VALID_EXTENSIONS = ['jpeg', 'JPEG', 'jpg', 'JPG', 'pdf', 'tiff', 'png']

ADDITIONAL_MARGIN_RATIO = {
    ILLUSTRATION_LABEL: 0,
    TEXT_LABEL: 0.5,
    TABLE_LABEL: 0,
    BASELINE_LABEL: 0.25,
    LINE_LABEL: 0.25, 
    COLUMN_LABEL: 0.25
}
LABEL_TO_DOCUMENT_CC_AREA_RATIO_THRESHOLD = {
    ILLUSTRATION_LABEL: 0.005,
    TEXT_LABEL: 0.00001,
    TABLE_LABEL: 0.01,
    BASELINE_LABEL: 0.0001,
    LINE_LABEL: 0.000015,
    COLUMN_LABEL: 0.000015
}
LABEL_TO_NAME = {
    ILLUSTRATION_LABEL: 'illustration',
    TEXT_LABEL: 'text',
    TABLE_LABEL: 'table',
    BASELINE_LABEL: 'cell',
    LINE_LABEL: 'line',
    COLUMN_LABEL: 'column'
}

class TableTranscriber:
    """
    Extract and transcribe tables from files in a given input_dir folder and save them in the provided output_dir.
    Supported input extensions are: jpg, png, tiff, pdf.
    Each input image should contain 0 or 1 table (no more than 1).
    This program is thought to work on a four branches network, with the following labels: 
    restricted_labels_1 = [1, 4, 6], restricted_labels_2 = [9], restricted_labels_3 = [14], restricted_labels_4 = [13]
    """

    def __init__(self, input_dir, output_dir, labels_to_extract=None, in_ext=VALID_EXTENSIONS, out_ext='jpg',
                 tag='four_branches_lines_columns', save_annotations=True, straight_bbox=False, add_margin=True, draw_margin=False):

        self.input_dir = coerce_to_path_and_check_exist(input_dir).absolute()
        self.files = get_files_from_dir(self.input_dir, valid_extensions=in_ext, recursive=True, sort=True)
        self.output_dir = coerce_to_path_and_create_dir(output_dir).absolute()
        self.out_extension = out_ext
        model_path = coerce_to_path_and_check_exist(MODELS_PATH / tag / MODEL_FILE)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model, (self.img_size, restricted_labels_1, restricted_labels_2, restricted_labels_3, restricted_labels_4, self.normalize) = load_model_from_path(
            model_path, device=self.device, attributes_to_return=['train_resolution', 'restricted_labels_1', 'restricted_labels_2', 'restricted_labels_3', 'restricted_labels_4', 'normalize'])
          
        self.model.eval()

        self.restricted_labels_1, self.restricted_labels_2, self.restricted_labels_3, self.restricted_labels_4 = sorted(restricted_labels_1), sorted(restricted_labels_2), sorted(restricted_labels_3), sorted(restricted_labels_4)
        self.restricted_labels = sorted(restricted_labels_1 + restricted_labels_2 + restricted_labels_3 + restricted_labels_4)
        
        self.labels_to_extract = [ILLUSTRATION_LABEL, TEXT_LABEL] if labels_to_extract is None else sorted(labels_to_extract)
        if not set(self.labels_to_extract).issubset(self.restricted_labels):
            raise ValueError('Incompatible `labels_to_extract` and `tag` arguments: '
                             f'model was trained using {self.restricted_labels} labels only')

        self.save_annotations = save_annotations
        self.straight_bbox = straight_bbox
        self.add_margin = add_margin
        self.draw_margin = add_margin and draw_margin

    def run(self):
        
        for filename in self.files:
                        
            try:
                imgs_with_names = self.get_images(filename)
            except (NotImplementedError, OSError) as e:
                imgs_with_names = []
            
            for img, name in imgs_with_names:
                #We first compute only the prediction of the second and third branch, which correspond, in our network four_branches_tables, to the table and column separators labels
                _, pred_2, pred_3, _ = self.predict(img)
           
                pred = pred_2
                label = TABLE_LABEL
                label_idx = self.restricted_labels_2.index(label) + 1
        
                mask_pred = cv2.resize((pred == label_idx).astype(np.uint8), img.size, interpolation=cv2.INTER_NEAREST)
                _, contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnt_areas = []

                for cnt in contours:
                    cnt_areas.append(cv2.contourArea(cnt))
                #If there is a table, with sufficient area, then:
                if len(cnt_areas) >= 1 and max(cnt_areas) / (img.size[0] * img.size[1]) >= LABEL_TO_DOCUMENT_CC_AREA_RATIO_THRESHOLD[label]:
                    #Rotation of the table according to the mean angle of all the columns separators
                    angle = self.rotate_image(img, pred_3, COLUMN_LABEL)
                    img = img.rotate(angle)

                    #After the rotation of the image, prediction of all the labels
                    pred_1, pred_2, pred_3, pred_4 = self.predict(img)
                    img_with_annotations = img.copy()
            
                    img_w, img_h = img.size
                    
                    #Extraction of all the different labels, with their corresponding coordinates (in order to recognize the structure of the table)
                    for label in self.labels_to_extract:
                        if label != BACKGROUND_LABEL:
                            if label in self.restricted_labels_1:
                                if(label == TEXT_LABEL):
                                    #Extraction of the cells (corresponding to the text label), with determination of the mean, and extreme coordinates of all the textual cells
                                    text_extracted_elements, x_text, y_text = self.extract(img, pred_1, label, img_with_annotations) 
                                    x_t_min = [np.min(x) for x in x_text]
                                    y_t_min = [np.min(y) for y in y_text]
                                    x_t_max = [np.max(x) for x in x_text]
                                    y_t_max = [np.max(y) for y in y_text]
                                    x_text = [np.mean(x) for x in x_text]
                                    y_text = [np.mean(y) for y in y_text]

                                else:
                                    extracted_elements, _, _ = self.extract(img, pred_1, label, img_with_annotations)  

                            elif label in self.restricted_labels_2:
                                #Extraction of the tables
                                extracted_elements, x_table, y_table = self.extract(img, pred_2, label, img_with_annotations)
                            elif label in self.restricted_labels_3:
                                #Extraction of the columns separators and determination of their coordinates
                                extracted_elements, x_coord, y_coord = self.extract(img, pred_3, label, img_with_annotations)
                                #Creation of the columns separators (1-dimensional) list definition, by taking the mean (on the x axis) of the x_coords of each column: projection of each column on the x-axis of the image, by reducing each column to its mean x-axis coordinates
                                x_coord = [np.mean(x) for x in x_coord]
                                #Deletion of too close columns separators, with a threshold of img width / 150
                                x_columns = self.delete_close(list(dict.fromkeys(np.sort(x_coord))), img_w/150)
                            elif label in self.restricted_labels_4:
                                #Extraction of the lines separators and determination of their coordinates
                                extracted_elements, x_coord, y_coord = self.extract(img, pred_4, label, img_with_annotations)
                                #Creation of the rows separators (1-dimensional) list definition, by taking the mean (on the y axis) of the y_coords of each row: projection of each row on the y-axis of the image, by reducing each row to its mean y-axis coordinates
                                y_coord = [np.mean(y) for y in y_coord]
                                #Deletion of too close rows separators, with a threshold of img height / 100
                                y_lines = self.delete_close(list(dict.fromkeys(np.sort(y_coord))), img_h/100)
                     
                    #Draw the grid separators on the annotated image            
                    draw = ImageDraw.Draw(img_with_annotations)
                    for y in (y_lines):
                        draw.line([(50, y), (100, y)], fill = 255, width = 10)
                    for x in (x_columns):
                        draw.line([(x, 50), (x, 100)], fill = 255, width = 10)

                    if self.save_annotations:
                        (self.output_dir / 'annotation').mkdir(exist_ok=True)
                        img_with_annotations.save(self.output_dir / 'annotation' / '{}_annotated.{}'
                                                  .format(name, self.out_extension))                 

                    self.create_tables(text_extracted_elements, x_text, y_text, x_t_min, y_t_min, x_t_max, y_t_max, y_lines, x_columns, x_table, y_table, self.output_dir / 'table_{}'.format(name), img.size, name)
    
    #Delete elements in list (for columns and rows separators) that are closer than a given threshold    
    def delete_close(self, coords, thresh):
        for i in range(len(coords)-1):
            if(coords[i+1] - coords[i] < thresh):
                coords[i + 1] = coords[i]
        return list(dict.fromkeys(coords))
    
    #Recognition of the table structure (determination of the index of each cell), post processing (cutting cells than span multiple columns), and creation of .html / .xsv tables transcriptions outputs
    def create_tables(self, text_extracted_elements, x_text, y_text, x_t_min, y_t_min, x_t_max, y_t_max, y_lines, x_columns, x_table, y_table, output_path, img_size, img_name):

        if output_path.exists():
            shutil.rmtree(output_path)
        output_path = coerce_to_path_and_create_dir(output_path)
        
        x_indices = {}
        y_indices = {}
                
        for i in range(len(text_extracted_elements)):
            
            if(np.array(x_table).min() <= x_text[i] <= np.array(x_table).max() and np.array(y_table).min() <= y_text[i] <= np.array(y_table).max()):
                w, h = text_extracted_elements[i].size
                #Determination of the minimum and maximum index of each text cell (in the table grid structure), by taking into account not the extreme coordinates of the cells, but a ponderate coordinate between the center of the cell and its min/max coordinates (in order not to cut cells than just span a few pixels more than the rows/columns separators)
                x_t_min_pond = (x_t_min[i] + x_text[i])/2
                x_t_max_pond = (x_t_max[i] + x_text[i])/2
                
                y_t_min_pond = (y_t_min[i] + y_text[i])/2
                y_t_max_pond = (y_t_max[i] + y_text[i])/2
                
                c_min_index = bisect(x_columns, x_t_min_pond)
                c_max_index = bisect(x_columns, x_t_max_pond)
                
                l_min_index = bisect(y_lines, y_t_min_pond)
                l_max_index = bisect(y_lines, y_t_max_pond)
                
                (output_path / 'cell_images').mkdir(exist_ok=True)
                                
                #Double loop to cut all the cells that span multiple rows or columns, and to save them according to their correct index
                #In reality, in the above lines, we don't cut the cells on the y-axis, because astronomical tables cells are generally well detected vertically, and trying to cut them according to this axis leads to an over-cutting of the cells
                if (c_min_index < c_max_index or l_min_index < l_max_index):
                    for c_index in np.arange(c_min_index, c_max_index+1):
                        for l_index in np.arange(l_min_index, l_max_index+1):
                            if(0 < c_index < len(x_columns) and 0 < l_index < len(y_lines)):
                                area = (max(0,x_columns[c_index-1] - x_t_min[i]),0,min(x_columns[c_index], x_t_max[i]) - x_t_min[i],h) 

                                x_min = max(x_columns[c_index-1], x_t_min[i])
                                x_max = min(x_columns[c_index], x_t_max[i])
                            
                                sub_element = text_extracted_elements[i].crop(area)
                                index = [l_index,c_index]
                                sub_element.save(output_path / 'cell_images' / '{}.{}'
                                            .format('({0},{1})'.format(index[0], index[1]), 'jpg'))                                         
                                x_indices['{},{}'.format(index[0], index[1])] = [x_min, x_max]
                                y_indices['{},{}'.format(index[0], index[1])] = [y_t_min[i], y_t_max[i]]
                #If the cells don't span multiple rows or columns, we index them according to their center coordinates, in relation to rows and columns separators
                else:
                    #bisect() functions determines the index of the closest element to x_text[i] or y_text[i] (i.e. the projected cells coordinates, on the x-axis and y-axis) in, respectively, the 1-dimensional lists x_columns and y_lines (which correspond the projected coordinates (on the x and y axis) of the columns and rows of the table) ; thanks to this bisect() functions, the index of each cell in the table can be easely computed
                    c_index = bisect(x_columns, x_text[i])
                    l_index = bisect(y_lines, y_text[i])
                    index = [l_index,c_index]
                    x_indices['{},{}'.format(index[0], index[1])] = [x_t_min[i], x_t_max[i]]
                    y_indices['{},{}'.format(index[0], index[1])] = [y_t_min[i], y_t_max[i]]
                    text_extracted_elements[i].save(output_path / 'cell_images' / '{}.{}'
                                                .format('({0},{1})'.format(index[0], index[1]), 'jpg')) 
                                                
        path = output_path / 'cell_images'
        #Call of the HTR line prediction model to transcribe all the cells
        os.system('CUDA_VISIBLE_DEVICES=0 python HTR/line_predictor.py --data_path {} --model_path HTR/trained_networks/medieval_numbers.pth --imgh 64'.format(path))
        
        #Creation of the corresponding HTML and XML files
        fileout_html = open(output_path / 'html_table_{}.html'.format(img_name), "w")
        fileout_xml = open(output_path / 'xml_transcript_{}.xml'.format(img_name), "w")
        
        xml = ""
        xml += """<?xml version="1.0" encoding="UTF-8"?>
        <alto xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
          xmlns="http://www.loc.gov/standards/alto/ns-v4#"
          xsi:schemaLocation="http://www.loc.gov/standards/alto/ns-v4# http://www.loc.gov/standards/alto/v4/alto-4-2.xsd">
          <Description>
            <MeasurementUnit>pixel</MeasurementUnit>
            <sourceImageInformation>
              <fileName></fileName>
          </sourceImageInformation>
        </Description>
        <Layout>
        <Page WIDTH="{0}"
              HEIGHT="{1}"
              PHYSICAL_IMG_NR="0"
              ID="eSc_dummypage_">
        <PrintSpace HPOS="0"
              VPOS="0"
              WIDTH="{0}"
              HEIGHT="{1}">
        <TextBlock ID="eSc_dummyblock_">\n""".format(img_size[0], img_size[1])
                
        html_table = f"<table border='1' id='{img_name}'>\n"
        for line in range(len(y_lines)+1):
            html_table += "<tr>\n"
            for column in range(len(x_columns)+1):
                file_name = (output_path / 'cell_images' / '{}.{}'.format('({0},{1})'.format(line, column), 'jpg')) 
                if Path(file_name).is_file():
                    [x_min, x_max] = x_indices['{},{}'.format(line,column)]
                    [y_min, y_max] = y_indices['{},{}'.format(line,column)]
                    
                    transcript = [line_file for line_file in open(path / 'predictions.txt') if '({},{})'.format(line, column) in line_file][0].strip().split()
                    if(len(transcript) >= 3):
                        transcript = transcript[2:]
                        transcript = [[int(s) for s in trans.split() if s.isdigit()] for trans in transcript]
                    else:
                        transcript = ''
                    html_table += "<td><img src='{0}' alt='Cell'><input type='text' id='cell' value ='{1}' width='30px' size ='1'></td>\n".format(file_name, ', '.join(map(str, [trans[0] for trans in transcript])))
                    
                    xml += '<TextLine ID=""\n'
                    xml += 'BASELINE="{} {} {} {}"\n'.format(x_min, y_max, x_max, y_max)
                    xml += 'HPOS="{}"\n'.format(x_min)
                    xml += 'VPOS="{}"\n'.format(y_min)
                    xml += 'WIDTH="{}"\n'.format(x_max - x_min)
                    xml += 'HEIGHT="{}">\n'.format(y_max - y_min)
                    xml += '<Shape><Polygon POINTS="{} {} {} {} {} {} {} {}"/></Shape>\n'.format(x_min, y_max, x_min, y_min, x_max, y_min, x_max, y_max)
                    xml += '<String CONTENT="{}"\n'.format(', '.join(map(str, [trans[0] for trans in transcript])))
                    xml += 'HPOS="{}"\n'.format(x_min)
                    xml += 'VPOS="{}"\n'.format(x_min)
                    xml += 'WIDTH="{}"\n'.format(x_max - x_min)
                    xml += 'HEIGHT="{}"></String>\n'.format(y_max - y_min)
                    xml += '</TextLine>\n'
                    
                else:
                    html_table += "<td><input type='text' id='cell' width='30px' size ='1'></td>\n"
            html_table += "</tr>\n"
        html_table += "</table>"
        xml += '</TextBlock>\n </PrintSpace>\n </Page>\n </Layout>\n </alto>'
        
        csv_fct = "const table=document.getElementById('"+img_name+"');let csv='';for(let i=0,row;row=table.rows[i];i++){for(let j=0,col;col=row.cells[j];j++){for(let k=0;k<col.children.length;k++){if(col.children[k].tagName==='INPUT'){csv+=col.children[k].value+','} } }csv+='\\n'}const blob=new Blob([csv],{type:'text/csv;charset=utf-8;'});if(navigator.msSaveBlob){navigator.msSaveBlob(blob,'table.csv')}else{const link=document.createElement('a');if(link.download!==undefined){const url=window.URL.createObjectURL(blob);link.setAttribute('href',url);link.setAttribute('download','table.csv');link.style.visibility='hidden';document.body.appendChild(link);link.click();document.body.removeChild(link)} }"
        html_table += '<button onclick="'+csv_fct+'">Export to CSV</button>'

        xml_fct = 'const blob=new Blob([`'+xml+'`],{type:"text/xml;charset=utf-8;"});if(navigator.msSaveBlob){navigator.msSaveBlob(blob,"table.xml")}else{const link=document.createElement("a");if(link.download!==undefined){const url=window.URL.createObjectURL(blob);link.setAttribute("href",url);link.setAttribute("download","table.xml");link.style.visibility="hidden";document.body.appendChild(link);link.click();document.body.removeChild(link)} }'
        html_table += "<button onclick='"+xml_fct+"'>Export to XML</button>"
        
        export_fct = "const table=document.getElementById('"+img_name+"');let txt='';for(let i=0,row; row=table.rows[i]; i++){for(let j=0,col; col=row.cells[j]; j++){for(let k=0;k<col.children.length;k++){if(col.children[k].tagName==='IMG'){txt+=col.children[k].src+' '}if(col.children[k].tagName==='INPUT'){txt+=col.children[k].value+'\\n'}}}}const blob=new Blob([txt],{type:'text/txt;charset=utf-8;'});if(navigator.msSaveBlob){navigator.msSaveBlob(blob,'table.txt')}else{const link=document.createElement('a');if(link.download!==undefined){const url=window.URL.createObjectURL(blob);link.setAttribute('href',url);link.setAttribute('download','table.txt');link.style.visibility='hidden';document.body.appendChild(link);link.click();document.body.removeChild(link)}}"
        html_table += '<button onclick="'+export_fct+'">Export to training data</button>'
               
        fileout_html.writelines(html_table)
        fileout_html.close()
        
        fileout_xml.writelines(xml)
        fileout_xml.close()

    #Outputs of the model according to the different final branches (and, consequently, the different labels)
    def predict(self, image):
        red_img = resize(image, size=self.img_size, keep_aspect_ratio=True)
        inp = np.array(red_img, dtype=np.float32) / 255
        if self.normalize:
            inp = ((inp - inp.mean(axis=(0, 1))) / (inp.std(axis=(0, 1)) + 10**-7))
        inp = torch.from_numpy(inp.transpose(2, 0, 1)).float().to(self.device)  # HWC -> CHW tensor
        with torch.no_grad():
            outputs = self.model(inp.reshape(1, *inp.shape))
            pred_1 = outputs[0][0].max(0)[1].cpu().numpy()
            pred_2 = outputs[1][0].max(0)[1].cpu().numpy()
            pred_3 = outputs[2][0].max(0)[1].cpu().numpy()
            pred_4 = outputs[3][0].max(0)[1].cpu().numpy()

        return pred_1, pred_2, pred_3, pred_4
        
    #Rotation of the image depending on the mean angle of all the column separators (corresponding to restricted_labels_3) of the table in the image
    def rotate_image(self, image, pred, label):
        label_idx = self.restricted_labels_3.index(label) + 1

        rotate_angle = 0
        total_rotate = []
                
        mask_pred = cv2.resize((pred == label_idx).astype(np.uint8), image.size, interpolation=cv2.INTER_NEAREST)
        _, contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            cnt_area = cv2.contourArea(cnt)

            if cnt_area / (image.size[0] * image.size[1]) >= LABEL_TO_DOCUMENT_CC_AREA_RATIO_THRESHOLD[label]:
                rect = cv2.minAreaRect(cnt)
                width, height, angle = int(rect[1][0]), int(rect[1][1]), rect[-1]
                if(label == COLUMN_LABEL):
                    if(angle < -45):
                        rotate_angle = 90 + angle
                    else:
                        rotate_angle = angle
                    
                    total_rotate.append(rotate_angle)
                    
        rotate_angle = statistics.mean(total_rotate)
        print('Rotation angle:' + str(rotate_angle))           
        return rotate_angle
        
    #Annotation functions
    def extract(self, image, pred, label, image_with_annotations):
        if label in self.restricted_labels_1:
            label_idx = self.restricted_labels_1.index(label) + 1
        elif label in self.restricted_labels_2:
            label_idx = self.restricted_labels_2.index(label) + 1
        elif label in self.restricted_labels_3:
            label_idx = self.restricted_labels_3.index(label) + 1
        else:
            label_idx = self.restricted_labels_4.index(label) + 1
        color = LABEL_TO_COLOR_MAPPING[label]
        
        mask_pred = cv2.resize((pred == label_idx).astype(np.uint8), image.size, interpolation=cv2.INTER_NEAREST)
        _, contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        y_coord = []
        x_coord = []
        y_coord_bis = []
        i=0
        for cnt in contours:
            
            cnt_area = cv2.contourArea(cnt)
            if self.save_annotations:
                draw = ImageDraw.Draw(image_with_annotations)
                draw.line(list(map(tuple, cnt.reshape(-1, 2).tolist())) + cnt[0][0].tolist(), fill=color, width=2)
   
            if cnt_area / (image.size[0] * image.size[1]) >= LABEL_TO_DOCUMENT_CC_AREA_RATIO_THRESHOLD[label]:
                if self.straight_bbox:
                    x, y, width, height = cv2.boundingRect(cnt)
                    if self.add_margin:
                        m = int(min(ADDITIONAL_MARGIN_RATIO[label] * width, ADDITIONAL_MARGIN_RATIO[label] * height))
                        bbox = np.asarray([[x-m, y-m], [x+width+m, y-m], [x+width+m, y+height+m], [x-m, y+height+m]])
                        bbox = np.clip(bbox, a_min=(0, 0), a_max=image.size)
                        margins = np.array([min(m, x), min(m, y), -min(image.size[0] - x - width, m),
                                            -min(image.size[1] - y - height, m)], dtype=np.int32)
                    else:
                        bbox = np.asarray([[x, y], [x+width, y], [x+width, y+height], [x, y+height]])
                    result_img = image.crop(tuple(bbox[0]) + tuple(bbox[2]))

                else:
                    rect = cv2.minAreaRect(cnt)
                    width, height, angle = int(rect[1][0]), int(rect[1][1]), rect[-1]
 
                    if self.add_margin:
                        m = int(min(ADDITIONAL_MARGIN_RATIO[label] * width, ADDITIONAL_MARGIN_RATIO[label] * height))
                        width, height = width + 2 * m, height + 2 * m
                        rect = (rect[0], (width, height), angle)
                        margins = np.array([m, m, -m, -m], dtype=np.int32)
                    bbox = np.int32(cv2.boxPoints(rect))
                    dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype=np.float32)
                    M = cv2.getPerspectiveTransform(bbox.astype(np.float32), dst_pts)
                    result_img = Image.fromarray(cv2.warpPerspective(np.array(image), M, (width, height)))

                    if angle < -45:
                        result_img = result_img.transpose(Image.ROTATE_90)

                if self.draw_margin:
                    width, height = result_img.size
                    lw = int(min([0.01 * width, 0.01 * height]))
                    draw = ImageDraw.Draw(result_img)
                    rect = np.array([0, 0, width, height], dtype=np.int32) + margins
                    draw.rectangle(rect.tolist(), fill=None, outline=(59, 178, 226), width=lw)

                results.append(result_img)
                y_coord.append(bbox[:, 1])
                x_coord.append(bbox[:, 0])

                if self.save_annotations:
                    lw = int(min([0.005 * image.size[0], 0.005 * image.size[1]]))
                    draw = ImageDraw.Draw(image_with_annotations)
                    #draw.line(list(map(tuple, bbox.tolist())) + [tuple(bbox[0])], fill=(0, 255, 0), width=lw)

        return results, x_coord, y_coord
    
    #Get all the images of a folder
    def get_images(self, filename):
            if filename.suffix in ['.jpeg', '.JPEG', '.jpg', '.JPG', '.png', '.tiff']:
                imgs, names = [Image.open(filename).convert('RGB')], [filename.stem]
            elif filename.suffix == '.pdf':
                self.print_and_log_info('Converting pdf to jpg')
                imgs = Pdf2Image.convert(filename)
                names = ['{}_p{}'.format(filename.stem, k + 1) for k in range(len(imgs))]
            else:
                raise NotImplementedError('"{}" extension is currently not supported'.format(filename.suffix[1:]))

            return zip(imgs, names)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transcribe images of tables into numerical tables')
    parser.add_argument('-i', '--input_dir', nargs='?', type=str, required=True, help='Input directory')
    parser.add_argument('-o', '--output_dir', nargs='?', type=str, required=True, help='Output directory')
    parser.add_argument('-t', '--tag', nargs='?', type=str, default='four_branches_tables', help='Model tag to use')
    parser.add_argument('-l', '--labels', nargs='+', type=int, default=[ILLUSTRATION_LABEL, TEXT_LABEL, TABLE_LABEL, LINE_LABEL, COLUMN_LABEL], help='Labels to extract')
    parser.add_argument('-s', '--save_annot', action='store_true', help='Whether to save annotations')
    parser.add_argument('-sb', '--straight_bbox', action='store_true', help='Use straight bounding box only to'
                        'fit connected components found, instead of rotated ones')
    parser.add_argument('-dm', '--draw_margin', action='store_true', help='Draw the margins added, for visual purposes')
    args = parser.parse_args()
    
    input_dir = coerce_to_path_and_check_exist(args.input_dir)
    transcribe = TableTranscriber(input_dir, args.output_dir, labels_to_extract=args.labels, tag=args.tag, save_annotations=args.save_annot, straight_bbox=args.straight_bbox,
                          draw_margin=args.draw_margin)
    transcribe.run()
    