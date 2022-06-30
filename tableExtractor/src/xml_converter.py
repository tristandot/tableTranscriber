import argparse
import json
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

from utils.constant import ILLUSTRATION_COLOR, TABLE_WORD_COLOR, PARAGRAPH_COLOR, LINE_COLOR, SEG_GROUND_TRUTH_FMT
from utils.logger import print_info, print_error

from utils import coerce_to_path_and_check_exist, coerce_to_path_and_create_dir, get_files_from_dir


class Xml2Image:
    """
    Convert XML annotated regions into image files. Arg `input_dir` must include original images as well as the
    corresponding XML files.
    """

    def __init__(self, input_dir, output_dir, out_ext='png', color=TABLE_WORD_COLOR,
                 verbose=True):
        self.input_dir = coerce_to_path_and_check_exist(input_dir)
        self.annotations_files = get_files_from_dir(self.input_dir, valid_extensions=['xml'], recursive=True, sort=True)
        self.annotations = self.load_xml(self.annotations_files)
        self.output_dir = coerce_to_path_and_create_dir(output_dir)
        self.out_ext = out_ext
        self.color = color
        self.mode = 'L' if isinstance(color, int) else 'RGB'
        self.background_color = 0 if isinstance(color, int) else (0, 0, 0)        
        self.verbose = verbose
        

    @staticmethod
    def load_xml(annotations_files):
        
        results = []
        for file in annotations_files:
            tree = ET.parse(file)
            root = tree.getroot()
            results.append(root)    
        
        return results

    def run(self):
        for annot in self.annotations:
            img, background = self.convert(annot)
            if img is not None:
                img.save(self.output_dir / SEG_GROUND_TRUTH_FMT.format(annot.attrib.get('filename').split('.')[0], self.out_ext))

    def convert(self, annot):
        name = annot.attrib.get('filename')
        name = name.replace('.png', '.jpg')
        name = name.replace('.TIFF', '.jpg')
        if self.verbose:
            print_info('Converting XML annotations for {}'.format(name))
        if not (self.input_dir / name).exists:
            print_error('Original image {} not found'.format(name))
            return None

        size = Image.open(self.input_dir / name).size
        background = Image.open(self.input_dir / name)
        img = Image.new(self.mode, size, color=self.background_color)
        draw = ImageDraw.Draw(img)
        
        
        for coords in annot.iter('Coords'):
            polygon = []
            points = coords.get('points')
            points = points.replace(',', ' ')
            points = points.split(' ')
            for i in range(0, len(points)): 
                points[i] = int(points[i]) 
            polygon = points
            draw.polygon(polygon, fill=self.color)
            points = polygon + polygon[0:2]
            points = list(zip(*[iter(points)]*2))
            horizontal_points = points[0:2]+points[2:4]
            vertical_points = points[1:3]+points[3:5]
            
            draw.line(points[0:2], fill=LINE_COLOR, width=int(size[1]*30/1280))
            draw.line(points[2:4], fill=LINE_COLOR, width=int(size[1]*30/1280))
            
            
            draw.line(points[1:3], fill=LINE_COLOR, width=int(size[0]*30/1280))
            draw.line(points[3:5], fill=LINE_COLOR, width=int(size[0]*30/1280))
            
        return img, background


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert VIA annotated regions into image files')
    parser.add_argument('-i', '--input_dir', nargs='?', type=str, required=True, help='Input directory where the'
                        'annotated images are, necessary to recover image sizes')
    parser.add_argument('-o', '--output_dir', nargs='?', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    conv = Xml2Image(args.input_dir, args.output_dir)
    conv.run()
