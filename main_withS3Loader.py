from PIL import Image
import os
import csv
import pandas as pd
from io import BytesIO
from torchdata.datapipes.iter import IterableWrapper, S3FileLoader

class CustomImageLoader(S3FileLoader):
    def __init__(self, urls, transforms, classes, labels_csv_path, bucket_name):
        super(CustomImageLoader, self).__init__(urls)
        self.labels = pd.read_csv(labels_csv_path)
        self.transforms = transforms
        self.classes = classes
        self.bucket_name = bucket_name

    def __transform_label__(self, growth_label, hole_label):
        return self.classes['growth'].index(growth_label), self.classes['hole'].index(hole_label)

    def load_and_check_images(self, folder=None):
        problematic_images = []
        for url in self.source_datapipe:
            if url.startswith("s3://"):  # Check if the URL is an S3 URL
                try:
                    image = Image.open(BytesIO(self.handler.s3_read(url)))
                    image.verify()
                    image.close()
                except (IOError, SyntaxError):
                    problematic_images.append((url, "S3"))
            else:  # Assume local file path
                try:
                    with Image.open(url) as image:
                        image.verify()
                except (IOError, SyntaxError):
                    problematic_images.append((url, "Local"))
        return problematic_images

def write_to_csv(file_name, data):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["URL", "Type"])
        for url, image_type in data:
            writer.writerow([url, image_type])

if __name__ == "__main__":
    folder = 's3://dev-model-training/dataset_21/training/'
    #folder = 's3://dev-model-training/dataset_21/validation/'
    folder = os.path.expanduser(folder)
    labels_csv_path = 's3://dev-model-training/dataset_21/training_labels.csv'
   # labels_csv_path = 's3://dev-model-training/dataset_21/validation_labels.csv'


    classes = {'growth': ['high_growth', 'low_growth', 'medium_growth'], 'hole': ['holes', 'no holes']}

    # Assuming you have a list of local file paths in the specified folder
    file_paths = [os.path.join(folder, file) for file in os.listdir(folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    loader = CustomImageLoader(file_paths, None, classes, labels_csv_path, None)

    # Load and check images
    problematic_images = loader.load_and_check_images()

    if len(problematic_images) < 1:
        print("All images are Ok")
    else:
        write_to_csv("problematic_images.csv", problematic_images)
