from nfnets import NFNet, MulticlassClassifier, CustomNfNet
from torchvision import transforms
import torch
from PIL import Image
import cv2
import boto3
from io import BytesIO
import time


def load_model(model_path, config, model_in_aws):
    nfnet = NFNet(
        variant=config['variant'],
        num_classes=config['num_classes'],
        alpha=config['alpha'],
        stochdepth_rate=config['stochdepth_rate'],
        se_ratio=0.5,
        activation=config['activation'])
    # model = MulticlassClassifier(nfnet)
    model = CustomNfNet(nfnet.stem, nfnet.body[0:6], stochdepth_rate=0.25, alpha=0.2, activation='gelu')
    # Initialize s3 bucket
    if model_in_aws:
        s3 = boto3.client('s3')
        pretrained_model_object = BytesIO(s3.get_object(Bucket=config['bucket'], Key=model_path)['Body'].read())
        weight = torch.load(pretrained_model_object) if config['device'] == 'cuda:0' \
            else (torch.load(pretrained_model_object, map_location='cpu'))

    else:
        weight = torch.load(model_path) if config['device'] == 'cuda:0' \
            else (torch.load(model_path, map_location='cpu'))
    model.load_state_dict(weight['model'])
    model.eval()
    return model


def build_eval_transformation(config):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config['model_width'], config['model_height']), transforms.InterpolationMode.BICUBIC),
        transforms.Normalize(mean=config['mean'], std=config['std']),
    ])


def predict(pil_image, transformation, model, device):
    image = transformation(pil_image).unsqueeze(0)
    if device == 'cuda:0':
        image = image.to(device)
        model.to(device)
    outputs = model(image)
    if isinstance(outputs, dict):
        probabilities = {}
        for i, out in enumerate(outputs):
            probabilities[out] = torch.nn.Softmax(dim=-1)(outputs[out])
    else:
        probabilities = torch.nn.Softmax(dim=-1)(outputs)
    return probabilities


def break_image(image, config):
    start_height = 0
    start_width = 0
    end_width = config['model_width']
    end_height = config['model_height']
    new_images = []
    while end_height != image.shape[0]:
        while end_width != image.shape[1]:
            new_image = image[start_height:end_height, start_width:end_width, :]
            if new_image.shape[0] == 320:
                new_images.append(Image.fromarray(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)))
            start_width += 320
            end_width += 320
            if end_width > image.shape[1]:
                dif = end_width - image.shape[1]
                start_width -= dif
                end_width = image.shape[1]
        start_width = 0
        end_width = 320
        start_height += 320
        end_height += 320
        if end_height > image.shape[0]:
            dif = end_height - image.shape[0]
            start_height -= dif
            end_height = image.shape[0]
    return new_images


def break_image_in_four_parts(image):
    width = image.shape[1]
    width_cutoff = width // 2
    l1 = image[:, :width_cutoff]
    l2 = image[:, width_cutoff:]
    height_cutoff_l1 = l1.shape[0] // 2
    height_cutoff_l2 = l2.shape[0] // 2
    first, second = l1[:height_cutoff_l1, :], l1[height_cutoff_l1:, :]
    third, fourth = l2[:height_cutoff_l2, :], l2[height_cutoff_l2:, :]
    return Image.fromarray(cv2.cvtColor(first, cv2.COLOR_BGR2RGB)), Image.fromarray(
        cv2.cvtColor(second, cv2.COLOR_BGR2RGB)), Image.fromarray(
        cv2.cvtColor(third, cv2.COLOR_BGR2RGB)), Image.fromarray(cv2.cvtColor(fourth, cv2.COLOR_BGR2RGB))


def postprocess_output(probabilities, classes):
    if isinstance(probabilities, dict):
        prediction_indexes = {}
        final_prediction = {}
        for key in probabilities.keys():
            _, prediction_indexes[key] = torch.max(probabilities[key], 1)
            final_prediction[key] = classes[key][prediction_indexes[key].item()]
        return final_prediction
    else:
        _, prediction_index = torch.max(probabilities, 1)
        return classes[prediction_index.item()]
