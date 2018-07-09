from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
# %matplotlib inline
from collections import Counter

url = "https://commondatastorage.googleapis.com/books1000/"
# last_precent_reported = None
data_root ='.'

def download_progress_hook(count, blockSize, totalSize):
  last_percent_reported  = None
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()

    last_percent_reported = percent
def maybe_download(filename,expected_bytes,force=False):
    dest_filename = os.path.join(data_root,filename)
    if force or not os.path.exists(dest_filename):
        print("attempting to download",filename)
        filename, _ = urlretrieve(url+filename, dest_filename, reporthook=download_progress_hook)
        print("\ndownload_complete")
    statsinfo = os.stat(dest_filename)
    if statsinfo.st_size == expected_bytes:
        print("found and verified",dest_filename)
    else:
        raise Exception('failed to verify '+dest_filename+" get it through browser")
    return dest_filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)


num_classes = 10
np.random.seed(133)
def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall(data_root)
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)


image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) -
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))

  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)

  return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)

'''

for filename in train_datasets:
    print(filename)
    with open(filename,"rb") as input_file:
        file_object = pickle.load(input_file)
        print(file_object[0])
        print(len(file_object))
        print(type(file_object[0]))
        for file_image in file_object:
            plt.imshow(file_image)
            plt.show

'''


def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes

  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class

        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise

  return valid_dataset, valid_labels, train_dataset, train_labels


train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


pickle_file = os.path.join(data_root, 'notMNIST.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise


statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

print(type(train_labels))
# def showing_img(trai_labels,trai_dataset):
#     for i in range(20):
#         print(trai_labels[i])
#         plt.imshow(trai_dataset[i])
#         plt.show()
# showing_img(train_labels,train_dataset)

def showing_freq(trai_labels):
    print(len(trai_labels))
    print(Counter(trai_labels.tolist()))



# showing_freq(train_labels)
# showing_freq(valid_labels)
# showing_freq(test_labels)

print(type(train_dataset))
print(train_dataset[0,:,:])

'''
def overlapping_counter(vali_dataset,train_dataset):
    counter = 0
    for i in range(len(vali_dataset)):
        if(vali_dataset[i] in train_dataset):
            counter += 1
            print(counter,i)
    print(counter)

overlapping_counter(valid_dataset,train_dataset)
overlapping_counter(test_dataset,train_dataset)

counter = 0
for i in range(len(train_dataset)):
    for j in range(len(valid_dataset)):
        # print((train_dataset[i]==valid_dataset[j]).all())
        if((train_dataset[i]==valid_dataset[j]).all()):
            # print(train_dataset[i]==valid_dataset[j])
            print(i,j)
            print(counter)
            counter+=1
print(counter)
'''
# print(valid_dataset[(train_dataset[:,None]==valid_dataset).all(axis=(28,10000)).any(0)])
# print valid_dataset[(valid_dataset==train_dataset[:,None]).all(axis=(28,10000)).any(0)]))


train_dataset.flags.writeable = False
valid_dataset.flags.writeable = False
test_dataset.flags.writeable = False

train_hash = [hash(e.tostring()) for e in train_dataset]
valid_hash = [hash(e.tostring()) for e in valid_dataset]
test_hash = [hash(e.tostring()) for e in test_dataset]

unique_train_hash = set(train_hash)
unique_valid_hash = set(valid_hash)
unique_test_hash = set(test_hash)
valid_overlap = unique_train_hash.intersection(set(valid_hash))
test_overlap = unique_train_hash.intersection(set(test_hash))

print(len(unique_train_hash))
print(len(set(valid_hash)))
print(len(set(test_hash)))
print('Duplicates inside training set: ', len(train_hash) - len(unique_train_hash))
print('Duplicates between training and validation: ', len(valid_overlap))
print('Duplicates between training and test: ', len(test_overlap))



from sklearn.linear_model import LogisticRegression

train_sample = train_dataset[:5000,:,:]
train_sample_labels = train_labels[:5000]

(samples, width, height) = train_sample.shape
train_sample = np.reshape(train_sample, (samples, width * height))

(samples, width, height) = test_dataset.shape
test_dataset_reshaped = np.reshape(test_dataset, (samples, width * height))

model = LogisticRegression(penalty='l2', C=1.0)
model.fit(train_sample, train_sample_labels)

train_score = model.score(train_sample, train_sample_labels)
test_score = model.score(test_dataset_reshaped, test_labels)
print('Training score = ', train_score)
print('Test score = ', test_score)

train_dataset_cross, train_labels_cross = make_arrays(len(unique_train_hash), image_size)
test_dataset_cross, test_labels_cross = make_arrays(len(unique_test_hash)-1176, image_size)
train_hashing_list = []
j = 0
for i in range(len(train_hash)):
    if(train_hash[i] not in train_hashing_list):
        train_dataset_cross[j,:,:] = train_dataset[i,:,:]
        train_labels_cross[j] = train_labels[i]
        j += 1
        train_hashing_list.append(train_hash[i])
j = 0
test_hashing_list = []
for i in range(len(test_hash)):
    if(test_hash[i] not in test_hashing_list and test_hash[i] not in train_hashing_list):
        test_dataset_cross[j,:,:] = test_dataset[i,:,:]
        test_labels_cross[j] = test_labels[i]
        j += 1
        test_hashing_list.append(test_hash[i])


from sklearn.linear_model import LogisticRegression

train_sample = train_dataset_cross[:5000,:,:]
train_sample_labels = train_labels_cross[:5000]

(samples, width, height) = train_sample.shape
train_sample = np.reshape(train_sample, (samples, width * height))

(samples, width, height) = test_dataset_cross.shape
test_dataset_reshaped = np.reshape(test_dataset_cross, (samples, width * height))

model = LogisticRegression(penalty='l2', C=1.0)
model.fit(train_sample, train_sample_labels)

train_score = model.score(train_sample, train_sample_labels)
test_score = model.score(test_dataset_reshaped, test_labels)
print('Training score = ', train_score)
print('Test score = ', test_score)
print(len(test_dataset_cross))
