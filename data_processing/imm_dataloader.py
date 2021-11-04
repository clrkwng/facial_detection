import torch
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data_processing.imm_dataset import *

# DataModule makes data reusable and easy to share.
class IMMDataModule(pl.LightningDataModule):
	# This code in Lightning makes sure that this method is called ONLY from one GPU.
	def __init__(self, batch_size, nose_keypoint_flag=True):
		super().__init__()
		self.batch_size = batch_size
		self.nose_keypoint_flag = nose_keypoint_flag
	
	def setup(self, stage):
		if self.nose_keypoint_flag:
			self.imm_dataset_train = NoseDataset(train_flag=True)
			self.imm_dataset_val = NoseDataset(train_flag=False)
		else:
			self.imm_dataset_train = FacialDataset(train_flag=True)
			self.imm_dataset_val = FacialDataset(train_flag=False)

	def train_dataloader(self):
		return DataLoader(dataset=self.imm_dataset_train, batch_size=self.batch_size, shuffle=True)#, num_workers=4)

	def val_dataloader(self):
		return DataLoader(dataset=self.imm_dataset_val, batch_size=self.batch_size, shuffle=False)#, num_workers=4)