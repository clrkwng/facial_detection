from comet_ml import Experiment
import os, os.path, argparse
from numpy.lib.function_base import extract
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger

from data_processing.imm_dataloader import *
from model.model_part1 import *

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', default=256, type=int,
		help='Batch size passed into dataloader.')
parser.add_argument('--num_epochs', default=100, type=int,
		help='Number of epochs to train the model.')
parser.add_argument('--lr', type=float, default=0.001,
		help='Learning rate used for optimizer.')
parser.add_argument('--momentum', type=float, default=0.9,
  	help='Momentum used for optimization.')
parser.add_argument('--optimizer', type=str, default='Adam',
		help='Optimizer used during learning.')

def main(args):
	comet_logger = CometLogger(
							api_key='5zqkkwKFbkhDgnFn7Alsby6py',
							workspace='clrkwng',
							project_name='facial_detection',
							experiment_name='initial_train',
	)
	BATCH_SIZE = args.batch_size
	NUM_EPOCHS = args.num_epochs
	LR = args.lr
	MOMENTUM = args.momentum
	OPTIMIZER = args.optimizer

	data_module = IMMDataModule(batch_size=BATCH_SIZE, nose_keypoint_flag=False)
	model = IMMClassifier(layers=[1,1,1,1],
												image_channels=1,
												num_epochs=NUM_EPOCHS,
												optimizer = OPTIMIZER,
												lr = LR,
												momentum = MOMENTUM,
												scheduler = None,
												save_path = 'pickled_files/best_imm_model.pt',
												nose_keypoint_flag = False,
												)

	trainer = pl.Trainer(gpus=1,
											 profiler='simple',
											 logger=comet_logger,
											 num_sanity_val_steps=0,
											 check_val_every_n_epoch=1,
											 max_epochs=NUM_EPOCHS)

	trainer.fit(model, data_module)

if __name__ == "__main__":
	if '--help' in sys.argv or '-h' in sys.argv:
		parser.print_help()
	else:
		argv = extract_args()
		args = parser.parse_args(argv)
		main(args)