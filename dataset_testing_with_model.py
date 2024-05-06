import torchvision
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch.cuda.amp as amp
import torch.nn as nn
from torchvision.transforms.transforms import Compose, Normalize, RandomResizedCrop, RandomAffine, RandomPerspective, \
    RandomApply, ToTensor, RandomHorizontalFlip, Resize
from torch.utils.data import DataLoader
import torch
from nfnets import NFNet, SGD_AGC, pretrained_nfnet, MulticlassClassifier, CustomNfNet
import time
from torch.utils.tensorboard import SummaryWriter
import boto3


def validate(model, validloader, criterion) -> torch.tensor:
    valid_loss = 0.0
    model.eval()  # Optional when not using Model Specific layer
    steps = 0
    for input in validloader:
        labels, data = input[1], input[0]
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        target = model(data)
        loss = criterion(target, labels)
        valid_loss += loss.item()
        steps += 1
    return valid_loss / steps


def train() -> None:
    s3 = boto3.client('s3')
    pretrained_model_object = s3.get_object(Bucket='dev-model-training', Key='pretrained_model/F0_haiku.npz')[
        'Body'].read()
    pretrained = pretrained_nfnet(
        model_object=pretrained_model_object,
        stochdepth_rate=0.25,
        alpha=0.2,
        activation='gelu'
    )

    # for name, param in pretrained.named_parameters():
    #     param.requires_grad = False
    # pretrained = torchvision.models.efficientnet_b2(weights=torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    # pretrained = torchvision.models.resnet34(weights='IMAGENET1K_V1')
    stem = pretrained.stem
    for name, param in stem.named_parameters():
        param.requires_grad = False
    body = pretrained.body[0:6]
    for name, param in body.named_parameters():
        param.requires_grad = False
    model = CustomNfNet(stem, body, stochdepth_rate=0.25, alpha=0.2, activation='gelu')
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    print(f"Total Ram Requirements {pytorch_total_params/1000000} MB")
    # for name, param in model.named_parameters():
    #     # if param.requires_grad and name.startswith(('stem', 'body.0', 'body.1', 'body.2', 'body.3', 'body.4')):
    #     if param.requires_grad:
    #         print(name)
    # Initialize Transformations
    train_transforms = Compose([RandomApply(torch.nn.ModuleList([RandomHorizontalFlip(),
                                                                 RandomResizedCrop(size=(1080, 1080)),
                                                                 RandomAffine(degrees=20, translate=(0.2, 0.3)),
                                                                 RandomPerspective(distortion_scale=0.3)]), p=0.3),
                                Resize((1080, 1080)),
                                ToTensor(),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
    valid_transforms = Compose([
        Resize((1080, 1080)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Initialize device
    if torch.cuda.is_available():
        device = 'cuda:0'
        print('cuda')
    else:
        device = 'cpu'

    # prepare dataloaders
    # train_dataset = torchvision.datasets.ImageFolder('../image_preparation/data/dataset_7/training',
    #                                                  transform=train_transforms)
    # validation_dataset = torchvision.datasets.ImageFolder('../image_preparation/data/dataset_7/validation',
    #                                                       transform=valid_transforms)
    train_dataset = torchvision.datasets.ImageFolder('./data/dataset_2/test_set',
                                                     transform=train_transforms)
    validation_dataset = torchvision.datasets.ImageFolder('./data/dataset_2/test_set',
                                                          transform=valid_transforms)
    dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=2)
    validation_loader = DataLoader(dataset=validation_dataset, shuffle=True, batch_size=2)

    # Initialize learning rate

    learning_rate = 0.01 * 2 / 256

    # Initialize cross entropy loss
    criterion = nn.CrossEntropyLoss()

    # Initialize gradient scaler
    scaler = amp.GradScaler()

    # stockdepth value options
    st_depths = [0.25]
    for st_depth in st_depths:
        # Move the model to device and data parallel
        # model = pretrained
        model.to(device)

        # Initialize optimizer
        optimizer = SGD_AGC(named_params=model.named_parameters(),
                            lr=learning_rate,
                            momentum=0.9,
                            clipping=0.1,
                            weight_decay=0.00002
                            )

        # Initialize loss values
        previous_validation_loss = 10000000000000.0
        current_validation_loss = 0.0

        # Initialize tensorboard
        writer = SummaryWriter()

        # Create checkpoints directory
        # Training loop for each epoch begins
        print(f"Training for stockdepth {st_depth}")
        # after
        # stem
        # torch.Size([2, 128, 77, 77])
        for epoch in range(100):
            model.train()
            running_loss = 0.0
            processed_imgs = 0
            correct_labels = 0
            epoch_time = time.time()
            step = 0

            # Training loop for each batch/ step begins
            for data in dataloader:
                targets = data[1]
                inputs = data[0]
                inputs = inputs.to(device)
                # print(data[0].shape)
                targets = targets.to(device)
                optimizer.zero_grad()
                with amp.autocast(enabled=False):
                    output = model(inputs)
                loss = criterion(output, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                processed_imgs += targets.size(0)
                _, predicted = torch.max(output, 1)
                correct_labels += (predicted == targets).sum().item()
                print(f"\rEpoch {epoch + 1}"
                      f"\tProcessed Images {processed_imgs}"
                      f"\tLoss {running_loss / (step + 1):6.4f}"
                      f"\tAcc {100.0 * correct_labels / processed_imgs:5.3f}%\t")
                step += 1
            # Calculate epapsed time for one epoch
            elapsed = time.time() - epoch_time
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'loss': loss
            },  'sizetest.pth')
            # Print epochs results
            print(f"Total steps {step} in epoch {epoch + 1}")
            print(f"Elapsed Total Time {elapsed:.3f}s, {elapsed / step:.3}s/step, {elapsed / processed_imgs:.3}s/img)")

            print('Training loss', running_loss / step)
            print('Training accuracy', 100.0 * correct_labels / processed_imgs)

            # Update tensorboard with epoch results
            writer.add_scalar("Training Loss", running_loss / step, epoch + 1)
            writer.add_scalar("Training Accuracy", 100.0 * correct_labels / processed_imgs, epoch + 1)

            # Validate the model
            # if epoch % 2 == 0:
            current_validation_loss = validate(model, validation_loader, criterion)
            print(f'Validation Loss of epoch {epoch + 1} is {current_validation_loss}')
            writer.add_scalar('Validation Loss', current_validation_loss, epoch + 1)
            previous_validation_loss = current_validation_loss

    writer.close()


if __name__ == '__main__':
    train()
