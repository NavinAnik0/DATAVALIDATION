import torch
from torchdata.datapipes.iter import S3FileLoader, Batcher, BucketBatcher
from typing import Iterator, Tuple
from torchdata.datapipes.utils import StreamWrapper
from io import BytesIO, StringIO
from PIL import Image
import boto3
import pandas as pd


class CustomImageFolder(S3FileLoader):
    def __init__(self, urls, transforms, classes):
        super(CustomImageFolder, self).__init__(urls)
        self.transforms = transforms
        self.classes = classes

    def __transform_label__(self, label):
        return self.classes.index(label)

    def __iter__(self) -> Iterator[Tuple[str, StreamWrapper]]:
        for url in self.source_datapipe:
            label = self.__transform_label__(url.split('/')[5])
            image = Image.open(BytesIO(self.handler.s3_read(url)))
            transformed_image = self.transforms(image)
            yield label, transformed_image

    def __len__(self) -> int:
        return len(self.source_datapipe)


class CustomImageLoader(S3FileLoader):
    def __init__(self, urls, transforms, classes, labels_csv_path, bucket_name):
        super(CustomImageLoader, self).__init__(urls)
        s3 = boto3.client("s3")
        csv_obj = s3.get_object(Bucket=bucket_name, Key=labels_csv_path)
        body = csv_obj['Body']
        csv_string = body.read().decode('utf-8')
        self.labels = pd.read_csv(StringIO(csv_string), header=0, index_col=False)
        self.transforms = transforms
        self.classes = classes

    def __transform_label__(self, growth_label, hole_label):
        return self.classes['growth'].index(growth_label), self.classes['hole'].index(hole_label)

    def __iter__(self) -> Iterator[Tuple[str, StreamWrapper]]:
        for url in self.source_datapipe:
            labels = self.labels[self.labels['image_path'] == url.split('/')[5]]
            growth_label, hole_label = self.__transform_label__(labels['growth'].values[0], labels['holes'].values[0])
            image = Image.open(BytesIO(self.handler.s3_read(url)))
            transformed_image = self.transforms(image)
            yield {'growth': growth_label, 'hole': hole_label}, transformed_image

    def __len__(self) -> int:
        return len(self.source_datapipe)


class CustomBatcher(Batcher):
    def __iter__(self):
        batch = []
        label_list = []
        image_list = []
        for label, image in self.datapipe:
            label_list.append(label)
            image_list.append(image)
            # batch.append(x)
            if len(label_list) == self.batch_size:
                yield torch.tensor(label_list, dtype=torch.long), \
                      torch.tensor(torch.stack(image_list), dtype=torch.float32)
                label_list = []
                image_list = []
        if len(label_list) > 0:
            if not self.drop_last:
                # yield self.wrapper_class(batch)
                yield torch.tensor(label_list, dtype=torch.long), \
                      torch.tensor(torch.stack(image_list), dtype=torch.float32)


def loader_collate(data):
    growth_label_list = []
    hole_label_list = []
    image_list = []
    for label, image in data[0]:
        growth_label_list.append(label['growth'])
        hole_label_list.append(label['hole'])
        image_list.append(image)
    return (torch.tensor(hole_label_list, dtype=torch.long), torch.tensor(growth_label_list, dtype=torch.long),
            torch.tensor(torch.stack(image_list), dtype=torch.float32))
