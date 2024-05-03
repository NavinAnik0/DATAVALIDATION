from PIL import Image
import os
import csv
import pandas as pd

class CustomImageLoader:
    def __init__(self, folder, transforms, classes, labels_csv_path):
        self.folder = folder
        self.transforms = transforms
        self.classes = classes
        self.labels = pd.read_csv(labels_csv_path)

    def __transform_label__(self, growth_label, hole_label):
        return self.classes['growth'].index(growth_label), self.classes['hole'].index(hole_label)

    def load_and_check_images(self):
        problematic_images = []
        for root, _, files in os.walk(self.folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    labels = self.labels[self.labels['image_path'] == file]
                    growth_label, hole_label = self.__transform_label__(labels['growth'].values[0], labels['holes'].values[0])
                    try:
                        with Image.open(file_path) as image:
                            print("Ok : ", file_path)
                            image.verify()
                    except (IOError, SyntaxError):
                        problematic_images.append((root, file))
                        print("Problem : ", file_path)
        return problematic_images

def write_to_csv(file_name, data):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Folder", "File"])
        for folder, file in data:
            writer.writerow([folder, file])

if __name__ == "__main__":
    folder = '~/Documents/Dataset/training'
    folder = os.path.expanduser(folder)
    labels_csv_path = '~/Documents/Dataset/training_labels.csv'

    classes = {'growth': ['high_growth', 'low_growth', 'medium_growth'], 'hole': ['holes', 'no holes']}

    loader = CustomImageLoader(folder, None, classes, labels_csv_path)

    problematic_images = loader.load_and_check_images()

    if len(problematic_images) < 1:
        print("All images are Ok")
    write_to_csv("problematic_images.csv", problematic_images)
