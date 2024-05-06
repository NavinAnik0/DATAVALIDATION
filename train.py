import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import yaml
import torch.cuda.amp as amp
import torch.nn as nn
from torchvision.transforms.transforms import Compose, Normalize, RandomResizedCrop, RandomAffine, RandomApply, \
    RandomPerspective, ToTensor, RandomHorizontalFlip, Resize
from torchdata.datapipes.iter import IterableWrapper
from torch.utils.data import DataLoader
from nfnets import (NFNet, SGD_AGC, pretrained_nfnet, MulticlassClassifier, CustomImageFolder, loader_collate,
                    CustomImageLoader, EarlyStopping, CustomNfNet)
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import boto3
from io import BytesIO
from datetime import date
import gc
from torch.profiler import profile, ProfilerActivity, record_function


def validate(model, validloader, calculate_loss, loss_func, device) -> torch.tensor:
    torch.cuda.empty_cache()
    valid_loss = 0.0
    model.eval()
    steps = 0
    for data in validloader:
        label_hole, label_growth, data = data[0], data[1], data[2]
        if torch.cuda.is_available():
            data, label_hole, label_growth = data.cuda(), label_hole.cuda(), label_growth.cuda()

        outputs = model(data)
        loss = calculate_loss(loss_func, outputs, label_hole, label_growth, device)
        valid_loss += loss.item()
        steps += 1
    return valid_loss / steps


def criterion(loss_func, outputs, target_hole, target_growth, device):
    losses = 0
    for i, key in enumerate(outputs):
        losses += loss_func(outputs[key], target_hole.to(device) if key == "hole" else target_growth.to(device))
    return losses


def train(config: dict) -> None:
    # Initialize s3 bucket
    s3 = boto3.client('s3')

    # Get pretrained model object from s3
    # pretrained_model_object = s3.get_object(Bucket=config['model_bucket_name'], Key=config['pretrained'])['Body'].read()
    # Initialize Transformations
    train_transforms = Compose([RandomApply(torch.nn.ModuleList([RandomHorizontalFlip(),
                                                                 # RandomResizedCrop(size=(1080, 1080)),
                                                                 # RandomAffine(degrees=20, translate=(0.2, 0.3)),
                                                                 # RandomPerspective(distortion_scale=0.2)
                                                                 ]), p=0.1),
                                Resize((1080, 1080)),
                                ToTensor(),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
    valid_transforms = Compose([
        Resize((1080, 1080)),
        ToTensor(),
        Normalize(mean=config['mean'], std=config['std']),
    ])
    # Initialize device
    if torch.cuda.is_available():
        device = 'cuda:0'
        print('cuda')
    else:
        print("CPU")
        device = config['device']

    # Prepare loaders to read data from s3 bucket
    training_dp_s3_urls = IterableWrapper([config['training_dataset']]).list_files_by_s3()
    validation_dp_s3_urls = IterableWrapper([config['validation_dataset']]).list_files_by_s3()
    training_sharded_s3_urls = training_dp_s3_urls.shuffle(buffer_size=config['training_dataset_len']).sharding_filter()
    validation_shared_s3_urls = validation_dp_s3_urls.shuffle(
        buffer_size=config['validation_dataset_len']).sharding_filter()
    # training_dp_s3_files = CustomImageFolder(training_sharded_s3_urls,
    #                                          train_transforms, config['classes']).batch(config['batch_size'])
    # validation_dp_s3_files = CustomImageFolder(validation_shared_s3_urls,
    #                                            valid_transforms, config['classes']).batch(config['batch_size'])

    training_dp_s3_files = CustomImageLoader(training_sharded_s3_urls, train_transforms, config['classes'],
                                             "dataset_21/training_labels.csv",
                                             "dev-model-training").batch(config['batch_size'])
    validation_dp_s3_files = CustomImageLoader(validation_shared_s3_urls, valid_transforms, config['classes'],
                                               "dataset_21/validation_labels.csv",
                                               "dev-model-training").batch(config['batch_size'])
    dataloader = DataLoader(dataset=training_dp_s3_files, collate_fn=loader_collate, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dp_s3_files, collate_fn=loader_collate, shuffle=True)

    print("data loader ready")
    print("Testing Training data set ...")
    step = 0;
    for data in dataloader:
        step = step+ 1
        print("Training step", step," .....")
        target_holes, target_growth,  inputs = data[0], data[1], data[2]
        print("..")

    print("Training data set Ok")

    print("Testing Validation data set ...")
    for data in validation_loader:
        print("Validation", step," .....")
        target_holes, target_growth, inputs = data[0], data[1], data[2]
        print("..")

    print("Validation data set Ok")

    # # Initialize learning rate
    # if config['scale_lr']:
    #     learning_rate = config['learning_rate'] * config['batch_size'] / 256
    # else:
    #     learning_rate = config['learning_rate']
    #
    # # Initialize clippers
    # if not config['do_clip']:
    #     config['clipping'] = None
    #
    # # Initialize early stopping
    # early_stopping = EarlyStopping(tolerance=5, min_delta=0.15)
    #
    # # Initialize cross entropy loss
    # loss_func = nn.CrossEntropyLoss()
    #
    # # Initialize gradient scaler
    # scaler = amp.GradScaler()
    #
    # # stockdepth value options
    # st_depths = config['stochdepth_rate']
    # for st_depth in st_depths:
    #     # Initialize model
    #     if config['pretrained'] is not None:
    #         pretrained = pretrained_nfnet(
    #             model_object=pretrained_model_object,
    #             # stochdepth_rate=config['stochdepth_rate'], # change for ndsw-128 test st depths
    #             stochdepth_rate=st_depth,
    #             alpha=config['alpha'],
    #             activation=config['activation']
    #         )
    #         stem = pretrained.stem
    #         for name, param in stem.named_parameters():
    #             param.requires_grad = False
    #         body = pretrained.body[0:6]
    #         for name, param in body.named_parameters():
    #             param.requires_grad = False
    #         model = CustomNfNet(stem, body, stochdepth_rate=st_depth, alpha=config['alpha'],
    #                             activation=config['activation'])
    #         del pretrained
    #         del body
    #         del stem
    #         gc.collect()
    #     else:
    #         nfnets = NFNet(
    #             num_classes=config['num_classes'],
    #             variant=config['variant'],
    #             stochdepth_rate=config['stochdepth_rate'],
    #             alpha=config['alpha'],
    #             se_ratio=config['se_ratio'],
    #             activation=config['activation']
    #         )
    #         model = MulticlassClassifier(nfnets)
    #
    #     # Initialize computations
    #     if config['use_fp16']:
    #         model.half()
    #
    #     # Move the model to device and data parallel
    #     model.to(device)
    #
    #     # Initialize optimizer
    #     optimizer = SGD_AGC(named_params=model.named_parameters(),
    #                         lr=learning_rate,
    #                         momentum=config['momentum'],
    #                         clipping=config['clipping'],
    #                         weight_decay=config['weight_decay'],
    #                         nesterov=config['nesterov']
    #                         )
    #
    #     # Find desired parameters and exclude them from weight decay and clipping
    #     for group in optimizer.param_groups:
    #         name = group['name']
    #
    #         if model.exclude_from_weight_decay(name):
    #             group['weight_decay'] = 0
    #
    #         if model.exclude_from_clipping(name):
    #             group['clipping'] = None
    #
    #     # Initialize loss values
    #     previous_validation_loss = 10000000000000.0
    #     current_validation_loss = 0.0
    #
    #     # Initialize tensorboard
    #     writer = SummaryWriter()
    #
    #     # Create checkpoints directory
    #     checkpoints_dir = 'checkpoints/experiment_' + str(config['experiment']) + '_' + 'st_depth_' + str(st_depth) + \
    #                       date.today().strftime("%d-%m-%Y")
    #     # END: NDSW-39
    #     # Training loop for each epoch begins
    #     print(f"Training for stockdepth {st_depth}")
    #     for epoch in range(config['epochs']):
    #         torch.cuda.empty_cache()
    #         gc.collect()
    #         model.train()
    #         running_loss = 0.0
    #         processed_imgs = 0
    #         correct_hole_labels = 0
    #         correct_growth_labels = 0
    #         epoch_time = time.time()
    #         step = 0
    #
    #         # Training loop for each batch/ step begins
    #         for data in dataloader:
    #             target_holes, target_growth,  inputs = data[0], data[1], data[2]
    #             inputs = inputs.half().to(device) if config['use_fp16'] else inputs.to(device)
    #             target_holes = target_holes.to(device)
    #             target_growth = target_growth.to(device)
    #             optimizer.zero_grad()
    #
    #             with amp.autocast(enabled=config['amp']):
    #                 outputs = model(inputs)
    #
    #             loss = criterion(loss_func, outputs, target_holes, target_growth, device)
    #
    #             # if (step + 1) % 2 == 0 or step == 2789:
    #             scaler.scale(loss).backward()
    #             scaler.step(optimizer)
    #             scaler.update()
    #             running_loss += loss.item()
    #             processed_imgs += target_holes.size(0)
    #             for i, out in enumerate(outputs):
    #                 _, predicted = torch.max(outputs[out], 1)
    #                 if out == "hole":
    #                     correct_hole_labels += (predicted == target_holes).sum().item()
    #                 else:
    #                     correct_growth_labels += (predicted == target_growth).sum().item()
    #             print(f"\rEpoch {epoch + 1}/{config['epochs']}"
    #                   f"\tProcessed Images {processed_imgs}"
    #                   f"\tLoss {running_loss / (step + 1):6.4f}"
    #                   f"\tGrowth Acc {100.0 * correct_growth_labels / processed_imgs:5.3f}"
    #                   f"\tHole Acc {100.0 * correct_hole_labels / processed_imgs:5.3f}%\t")
    #             step += 1
    #             torch.cuda.empty_cache()
    #         # Calculate epapsed time for one epoch
    #         elapsed = time.time() - epoch_time
    #         # Print epochs results
    #         print(f"Total steps {step} in epoch {epoch + 1}")
    #         print(f"Elapsed Total Time {elapsed:.3f}s, {elapsed / step:.3}s/step, {elapsed / processed_imgs:.3}s/img)")
    #
    #         print('Training loss', running_loss / step)
    #         print('Training accuracy', 100.0 * correct_growth_labels / processed_imgs,
    #               100.0 * correct_hole_labels / processed_imgs)
    #
    #         # Update tensorboard with epoch results
    #         writer.add_scalar("Training Loss", running_loss / step, epoch + 1)
    #         writer.add_scalar("Training Growth Accuracy", 100.0 * correct_growth_labels / processed_imgs, epoch + 1)
    #         writer.add_scalar("Training Hole Accuracy", 100.0 * correct_hole_labels / processed_imgs, epoch + 1)
    #
    #         # START: NDSW-39
    #         # Validate the model
    #         # if epoch % 2 == 0:
    #         torch.cuda.empty_cache()
    #
    #         current_validation_loss = validate(model, validation_loader, criterion, loss_func, device)
    #         print(f'Validation Loss of epoch {epoch + 1} is {current_validation_loss}')
    #         writer.add_scalar('Validation Loss', current_validation_loss, epoch + 1)
    #         # Save the checkpoint
    #         if current_validation_loss < previous_validation_loss:
    #             cp_path = checkpoints_dir + "/epoch_" + str(epoch + 1) + ".pth"
    #             buffer = BytesIO()
    #             torch.save({
    #                 'epoch': epoch,
    #                 'model': model.state_dict(),
    #                 'optim': optimizer.state_dict(),
    #                 'loss': loss
    #             }, buffer)
    #             buffer.seek(0)
    #             s3.upload_fileobj(
    #                 buffer,
    #                 config['bucket_name'],
    #                 cp_path,
    #             )
    #             print(f"Saved checkpoint to {str(cp_path)}")
    #         previous_validation_loss = current_validation_loss
    #         # early stopping
    #         early_stopping(running_loss, current_validation_loss)
    #         if early_stopping.early_stop:
    #             print("We are early stopping at epoch:", epoch + 1)
    #             break
    #         # END: NDSW-39
    #
    # writer.close()


if __name__ == '__main__':
    with open('config_files/training.yaml') as file:
        config = yaml.safe_load(file)
    print("Started...")
    train(config=config)
