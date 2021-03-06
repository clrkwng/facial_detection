"""
This is a custom dataset to use on top of the IMM Face Database.
"""

import glob, torch, sys
# According to documentation, need to set seed for torch random generator.
torch.manual_seed(17)

import numpy as np
import torchvision.transforms as transforms

from PIL import Image, ImageOps
from torch.utils.data import Dataset

from utils.utils import *

# sys.path.insert(0, '../utils/')
# from utils import *
# sys.path.pop(0)

DATA_MEAN = np.array([0.24463636])
DATA_STD = np.array([0.13665744])

class NoseDataset(Dataset):
	def __init__(self, train_flag, folder_path='data/imm_face_db/', \
		nose_keypoint_dict_pkl_path='/home/kman/explorations/facial_detection/pickled_files/nose_keypoint_dict.pkl'):

		# TODO: Split this into train/val dicts.
		self.nose_keypoint_dict = load_pickle(nose_keypoint_dict_pkl_path)

		if train_flag:
			self.im_file_names = glob.glob(f"{folder_path}train/*.jpg")
		else:
			self.im_file_names = glob.glob(f"{folder_path}validation/*.jpg")

		self.data_len = len(self.im_file_names)
		self.transform = transforms.Compose([
										 transforms.ToTensor(),
										 transforms.Resize(60), # TODO: Problem might be using a different mean/std for the resized data.
										 transforms.Normalize(mean=DATA_MEAN, std=DATA_STD)])

	def __len__(self):
		return self.data_len

	def __getitem__(self, index):
		if torch.is_tensor(index):
			index = index.tolist()

		im_file = self.im_file_names[index]
		im = Image.open(im_file).convert('RGB')
		im = ImageOps.grayscale(im)
		im = np.asarray(im).copy()
		im = self.transform(im)

		im_name = im_file.split('/')[-1].split('.')[0]
		label = self.nose_keypoint_dict[im_name]

		return (im, label)

class FacialDataset(Dataset):
	def __init__(self, train_flag, folder_path='data/imm_face_db/', \
		keypoint_dict_pkl_path='/home/kman/explorations/facial_detection/pickled_files/facial_keypoint_dict.pkl'):

		# TODO: Split this into train/val dicts.
		self.keypoint_dict = load_pickle(keypoint_dict_pkl_path)
		self.train_flag = train_flag

		if self.train_flag:
			self.im_file_names = glob.glob(f"{folder_path}train/*.jpg")
		else:
			self.im_file_names = glob.glob(f"{folder_path}validation/*.jpg")

		self.data_len = len(self.im_file_names)

		if self.train_flag:
			self.transform = transforms.Compose([
											 transforms.ToPILImage(),
											 transforms.ColorJitter(),
											 transforms.ToTensor(),
											 transforms.Resize(120), # TODO: Problem might be using a different mean/std for the resized data.
										 	 transforms.Normalize(mean=DATA_MEAN, std=DATA_STD)])
		else:
			self.transform = transforms.Compose([
											 transforms.ToTensor(),
											 transforms.Resize(120), # TODO: Problem might be using a different mean/std for the resized data.
										 	 transforms.Normalize(mean=DATA_MEAN, std=DATA_STD)])

	def __len__(self):
		return self.data_len

	def __getitem__(self, index):
		if torch.is_tensor(index):
			index = index.tolist()

		im_file = self.im_file_names[index]
		im = Image.open(im_file).convert('RGB')
		im = ImageOps.grayscale(im)
		im = np.asarray(im).copy()
		im = self.transform(im)

		im_name = im_file.split('/')[-1].split('.')[0]
		label = self.keypoint_dict[im_name]
		return (im, label)