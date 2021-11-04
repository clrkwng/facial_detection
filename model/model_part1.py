from comet_ml import Experiment

import math, json, torch, sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import pytorch_lightning as pl
import torchvision.models as models

from utils.utils import *

# This is the residual block used in ResNets.
class ResBlock(pl.LightningModule):
  def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(intermediate_channels)
    self.conv2 = nn.Conv2d(intermediate_channels,
                            intermediate_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False)
    self.bn2 = nn.BatchNorm2d(intermediate_channels)
    self.relu = nn.ReLU(inplace=True)
    self.identity_downsample = identity_downsample
    self.stride = stride

  def forward(self, x):
    identity = x.clone()

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)

    if self.identity_downsample is not None:
      identity = self.identity_downsample(identity)

    x += identity
    x = self.relu(x)
    return x

class IMMClassifier(pl.LightningModule):
  def __init__(self, layers, image_channels, num_epochs, optimizer, lr, momentum, scheduler, save_path, nose_keypoint_flag=True):
    super().__init__()

    self.optimizer = optimizer
    self.lr = lr
    self.momentum = momentum
    self.num_epochs = num_epochs
    self.scheduler = scheduler

    self.model_save_path = save_path
    self.nose_keypoint_flag = nose_keypoint_flag

    self.in_channels = 64
    self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    # Mini ResNet architecture in these 4 lines below.
    self.layer1 = self._make_layer(layers[0], intermediate_channels=64, stride=1)
    self.layer2 = self._make_layer(layers[1], intermediate_channels=128, stride=2)
    self.layer3 = self._make_layer(layers[2], intermediate_channels=256, stride=2)
    self.layer4 = self._make_layer(layers[3], intermediate_channels=512, stride=2)

    self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    if self.nose_keypoint_flag:
      self.pixel_layers = nn.Sequential(nn.Linear(512, 256),\
                                        nn.ReLU(), \
                                        nn.Linear(256, 32),\
                                        nn.ReLU(),\
                                        nn.Linear(32, 8),\
                                        nn.ReLU(),\
                                        nn.Linear(8, 2))
    else:
      self.pixel_layers = nn.Sequential(nn.Linear(512, 256),\
                                        nn.ReLU(), \
                                        nn.Linear(256, 116))
    self.best_val_loss = 1e10

  def _make_layer(self, num_residual_blocks, intermediate_channels, stride):
    identity_downsample = None
    layers = []

    # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
    # we need to adapt the Identity (skip connection) so it will be able to be added
    # to the layer that's ahead
    if stride != 1 or self.in_channels != intermediate_channels:
      identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels,
                                                    intermediate_channels,
                                                    kernel_size=1,
                                                    stride=stride,
                                                    bias=False),
                                          nn.BatchNorm2d(intermediate_channels),)
    layers.append(ResBlock(self.in_channels, intermediate_channels, identity_downsample, stride))
    self.in_channels = intermediate_channels

    # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
    # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
    # and also same amount of channels.
    for i in range(num_residual_blocks - 1):
      layers.append(ResBlock(self.in_channels, intermediate_channels))

    return nn.Sequential(*layers)

  def configure_optimizers(self):
    # Pass in self.parameters(), since the LightningModule IS the model.
    if self.optimizer == "SGD":
      optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
    elif self.optimizer == "Adam":
      optimizer = optim.Adam(self.parameters(), lr=self.lr)
    else:
      print("Optimizer must be either SGD or Adam.")
      sys.exit(-1)

    if self.scheduler == None:
      return optimizer
    elif self.scheduler == "StepLR":
      scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[9], gamma=0.9)
    elif self.scheduler == "CosineAnnealingLR":
      scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.num_epochs)
    else:
      print("Scheduler must be StepLR/CosineAnnealingLR/None.")
      sys.exit(-1)
    return [optimizer], [scheduler]

  def mse_loss(self, preds, labels):
    preds = preds.float()
    labels = labels.float()
    criterion = torch.nn.MSELoss()
    return criterion(preds, labels)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.reshape(x.shape[0], -1)

    preds = self.pixel_layers(x)
    return preds

  def training_step(self, train_batch, batch_idx):
    inputs, lbls = train_batch
    preds = self.forward(inputs)

    loss = self.mse_loss(preds, lbls)
    # self.log('train_loss', loss)

    return loss

  def validation_step(self, val_batch, batch_idx):
    with self.logger.experiment.validate():
      inputs, lbls = val_batch
      preds = self.forward(inputs)
      
      loss = self.mse_loss(preds, lbls)
      self.log('val_loss', loss)

      if loss < self.best_val_loss:
        self.best_val_loss = loss
        torch.save(self.state_dict(), self.model_save_path)