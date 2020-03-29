import wget
import os
import zipfile


# Downloading the horse and human dataset
URL = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"
wget.download(URL, './Images')

# Extracting the data from zip files
local_zip = './Images/horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./Images')
zip_ref.close()

# pointers to the training directories
train_horse_dir = os.path.join('/Images/horses')
train_human_dir = os.path.join('/Images/humans')

#  Stats about the images present
print('total training horse images:', len(os.listdir(train_horse_dir)))
print('total training human images:', len(os.listdir(train_human_dir)))

