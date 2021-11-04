import ipdb, cv2, glob, pickle, sys
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
from skimage.transform import resize
from tqdm import tqdm

def load_pickle(file):
  """
  Given path to a pickle file, load and return the pickled object.
  """
  with open(file, 'rb') as f:
    obj = pickle.load(f)
  return obj

def save_pickle(obj, file):
  """
  Given object to be pickled and file to save to, pickle the object.
  """
  with open(file, 'wb') as f:
    pickle.dump(obj, f)

def collect_nose_keypoints(root_dir):
  asf_files = glob.glob(f"{root_dir}train/*.asf") \
            + glob.glob(f"{root_dir}validation/*.asf")
            
  nose_keypoint_dict = {}

  for f in sorted(asf_files):
    file = open(f)
    points = file.readlines()[16:74]
    landmark = []

    for point in points:
      x,y = point.split('\t')[2:4]
      landmark.append([float(x), float(y)])

    # The nose keypoint. To plot it on top of the image,
    # the coordinates to use are (nose_keypoint[0] * im.shape[1], nose_keypoint[1] * im.shape[0]).
    nose_keypoint = np.array(landmark).astype('float32')[-6]
    datapt_name = f.split('/')[-1].split('.')[0]
    nose_keypoint_dict[datapt_name] = nose_keypoint

  save_pickle(nose_keypoint_dict, "pickled_files/nose_keypoint_dict.pkl")
  return nose_keypoint_dict

def collect_facial_keypoints(root_dir):
  asf_files = glob.glob(f"{root_dir}train/*.asf") \
            + glob.glob(f"{root_dir}validation/*.asf")

  facial_keypoint_dict = {}

  for f in sorted(asf_files):
    file = open(f)
    points = file.readlines()[16:74]
    landmark = []

    for point in points:
      x,y = point.split('\t')[2:4]
      landmark.append([float(x), float(y)])

    datapt_name = f.split('/')[-1].split('.')[0]
    facial_keypoint_dict[datapt_name] = np.array(landmark).astype('float32').flatten()

  save_pickle(facial_keypoint_dict, "pickled_files/facial_keypoint_dict.pkl")
  return facial_keypoint_dict

def calc_dataset_mean_std(resize_shape, root_dir='data/imm_face_db/'):
  im_file_names = glob.glob(f"{root_dir}train/*.jpg") \
            + glob.glob(f"{root_dir}validation/*.jpg")

  pixel_num = 0
  channel_sum = np.zeros(1)
  channel_sum_squared = np.zeros(1)

  for img_path in tqdm(im_file_names):
    im = Image.open(img_path).convert('RGB')
    im = ImageOps.grayscale(im)
    im = np.asarray(im).copy()
    im = im / 255.0
    im = resize(im, resize_shape, anti_aliasing=True)
    pixel_num += (im.size) # Increment number of pixels per channel.
    channel_sum += np.sum(im, axis=(0,1))
    channel_sum_squared += np.sum(np.square(im), axis=(0,1))
  
  rgb_mean = channel_sum / pixel_num
  rgb_std = np.sqrt(channel_sum_squared / pixel_num - np.square(rgb_mean))

  save_pickle(rgb_mean, 'pickled_files/data_mean.pkl')
  save_pickle(rgb_std, 'pickled_files/data_std.pkl')
  return (rgb_mean, rgb_std)

def extract_args(input_argv=None):
  """
  Pull out command-line arguments after "--". Blender ignores command-line flags
  after --, so this lets us forward command line arguments from the blender
  invocation to our own script.
  """
  if input_argv is None:
    input_argv = sys.argv
  output_argv = []
  if '--' in input_argv:
    idx = input_argv.index('--')
    output_argv = input_argv[(idx + 1):]
  return output_argv


def parse_args(parser, argv=None):
  return parser.parse_args(extract_args(argv))