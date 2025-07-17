import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image
import json
import yaml
from pathlib import Path
import argparse

class YOLOMultilabelDatasetBuilder:
    def __init__(self, annotations_folder, images_folder, output_folder="dataset"):
        self.annotations_folder = Path(annotations_folder)
        self.images_folder = Path(images_folder)
        self.output_folder = Path(output_folder)
        self.class_names = {}
        self.dataset = []

        self.output_folder.mkdir(parents=True, exist_ok=True)

    def get_image_extensions(self):
        """Find all valid image file extensions in the images folder."""
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        exts = set()
        for file in self.images_folder.iterdir():
            if file.is_file() and file.suffix.lower() in valid_exts:
                exts.add(file.suffix.lower())
        return list(exts)

    def load_class_names(self, data_yaml_file=None, classes_file=None):
        if data_yaml_file and os.path.exists(data_yaml_file):
            try:
                with open(data_yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                if 'names' in data:
                    if isinstance(data['names'], list):
                        for i, name in enumerate(data['names']):
                            self.class_names[i] = name
                    elif isinstance(data['names'], dict):
                        for class_id, name in data['names'].items():
                            self.class_names[int(class_id)] = name
                    print(f"Loaded class names from data.yaml: {list(self.class_names.values())}")
                    return
            except Exception as e:
                print(f"Error reading data.yaml: {e}")
        if classes_file and os.path.exists(classes_file):
            try:
                with open(classes_file, 'r') as f:
                    for i, line in enumerate(f):
                        self.class_names[i] = line.strip()
                print(f"Loaded class names from classes.txt: {list(self.class_names.values())}")
                return
            except Exception as e:
                print(f"Error reading classes.txt: {e}")
        # Fallback: parse annotation files for all class IDs
        all_classes = set()
        for ann_file in self.annotations_folder.glob("*.txt"):
            try:
                with open(ann_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.strip().split()[0])
                            all_classes.add(class_id)
            except Exception as e:
                print(f"Error reading {ann_file}: {e}")
        for class_id in sorted(all_classes):
            self.class_names[class_id] = f"class_{class_id}"
        print(f"Auto-generated classes: {list(self.class_names.values())}")

    def find_matching_image(self, annotation_file):
        base_name = annotation_file.stem
        image_extensions = self.get_image_extensions()
        for ext in image_extensions:
            image_path = self.images_folder / f"{base_name}{ext}"
            if image_path.exists():
                return image_path
        return None

    def parse_yolo_annotation(self, annotation_file):
        classes_in_image = set()
        try:
            with open(annotation_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        classes_in_image.add(class_id)
        except Exception as e:
            print(f"Error parsing {annotation_file}: {e}")
        return classes_in_image

    def create_multilabel_vector(self, classes_in_image):
        num_classes = len(self.class_names)
        vector = [0] * num_classes
        for class_id in classes_in_image:
            if class_id in self.class_names:
                vector[class_id] = 1
        return vector

    def process_dataset(self):
        annotation_files = list(self.annotations_folder.glob("*.txt"))
        if not annotation_files:
            print("No annotation files found.")
            return
        print(f"Processing {len(annotation_files)} annotation files...")
        for ann_file in annotation_files:
            image_path = self.find_matching_image(ann_file)
            if image_path is None:
                print(f"No image for {ann_file.name}")
                continue
            classes = self.parse_yolo_annotation(ann_file)
            if not classes:
                print(f"No valid annotations in {ann_file.name}")
                continue
            multilabel_vector = self.create_multilabel_vector(classes)
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                self.dataset.append({
                    'image_path': str(image_path),
                    'image_name': image_path.name,
                    'annotation_path': str(ann_file),
                    'width': width,
                    'height': height,
                    'classes': list(classes),
                    'multilabel_vector': multilabel_vector,
                    'num_objects': len(classes)
                })
            except Exception as e:
                print(f"Error reading image {image_path}: {e}")
        print(f"Processed {len(self.dataset)} image-annotation pairs.")

    def save_dataset(self, format_type='csv'):
        if not self.dataset:
            print("Dataset is empty. Nothing to save.")
            return
        df_data = []
        for item in self.dataset:
            row = {
                'image_path': item['image_path'],
                'image_name': item['image_name'],
                'annotation_path': item['annotation_path'],
                'width': item['width'],
                'height': item['height'],
                'classes': ','.join(map(str, item['classes'])),
                'num_objects': item['num_objects']
            }
            for i, name in self.class_names.items():
                row[f'has_{name}'] = item['multilabel_vector'][i]
            df_data.append(row)
        df = pd.DataFrame(df_data)
        if format_type in ['csv', 'both', 'all']:
            csv_path = self.output_folder / 'multilabel_dataset.csv'
            df.to_csv(csv_path, index=False)
            print(f"Saved to {csv_path}")
        if format_type in ['json', 'both', 'all']:
            json_path = self.output_folder / 'multilabel_dataset.json'
            with open(json_path, 'w') as f:
                json.dump(self.dataset, f, indent=2)
            print(f"Saved to {json_path}")
        self.save_training_file()
        self.save_statistics()
        class_json = self.output_folder / 'class_names.json'
        with open(class_json, 'w') as f:
            json.dump(self.class_names, f, indent=2)
        print(f"Saved class mapping to {class_json}")

    def save_training_file(self):
        training_path = self.output_folder / 'training_data.txt'
        with open(training_path, 'w') as f:
            for item in self.dataset:
                vector = ' '.join(map(str, item['multilabel_vector']))
                f.write(f"{item['image_name']} {vector}\n")
        print(f"Training data saved to {training_path}")
        header_path = self.output_folder / 'training_data_format.txt'
        with open(header_path, 'w') as f:
            f.write("Training Data Format:\n")
            f.write("Each line: image_name class0 class1 ... classN\n\n")
            for i, name in self.class_names.items():
                f.write(f"Position {i}: {name}\n")
        print(f"Saved training header to {header_path}")

    def save_statistics(self):
        stats = {
            'total_images': len(self.dataset),
            'total_classes': len(self.class_names),
            'class_names': self.class_names,
            'class_distribution': {},
            'average_objects_per_image': float(np.mean([item['num_objects'] for item in self.dataset])) if self.dataset else 0,
            'max_objects_per_image': max([item['num_objects'] for item in self.dataset]) if self.dataset else 0,
            'min_objects_per_image': min([item['num_objects'] for item in self.dataset]) if self.dataset else 0
        }
        for class_id, name in self.class_names.items():
            count = sum(1 for item in self.dataset if item['multilabel_vector'][class_id] == 1)
            stats['class_distribution'][name] = count
        stats_path = self.output_folder / 'dataset_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved statistics to {stats_path}")
        print("\n=== Statistics ===")
        print(f"Images: {stats['total_images']}, Classes: {stats['total_classes']}")
        for name, count in stats['class_distribution'].items():
            print(f"  {name}: {count}")

    def copy_dataset_images(self):
        """
        Copy all images with annotations into an 'images' subfolder in the output folder.
        """
        images_out_folder = self.output_folder / "images"
        images_out_folder.mkdir(parents=True, exist_ok=True)
        count = 0
        for item in self.dataset:
            src = Path(item['image_path'])
            dst = images_out_folder / src.name
            try:
                shutil.copy2(src, dst)
                count += 1
            except Exception as e:
                print(f"Failed to copy {src} to {dst}: {e}")
        print(f"Copied {count} images to {images_out_folder}")

def main():
    parser = argparse.ArgumentParser(description='YOLO to Multi-label Converter')
    parser.add_argument('--annotations', '-a', default="input_data/labels", help='Annotation folder')
    parser.add_argument('--images', '-i', default="input_data/images1", help='Image folder')
    parser.add_argument('--output', '-o', default='dataset', help='Output folder')
    parser.add_argument('--data-yaml', '-d', default="input_data/data.yaml", help='data.yaml path')
    parser.add_argument('--classes', '-c', help='classes.txt path (fallback)')
    parser.add_argument('--format', '-f', choices=['csv', 'json', 'both', 'all'], default='all', help='Save format')

    args = parser.parse_args()

    builder = YOLOMultilabelDatasetBuilder(
        annotations_folder=args.annotations,
        images_folder=args.images,
        output_folder=args.output
    )

    builder.load_class_names(args.data_yaml, args.classes)
    builder.process_dataset()
    builder.save_dataset(args.format)
    builder.copy_dataset_images()

if __name__ == "__main__":
    main()
