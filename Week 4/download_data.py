import wget
import zipfile

# For training Data
# Downloading the horse and human dataset
URL = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"
wget.download(URL, './Images/Training')

# Extracting the data from zip files
local_zip = './Images/Training/horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./Images/Training')
zip_ref.close()

# For validation
# Downloading the horse and human dataset
URL = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip"
wget.download(URL, './Images/Validation')

# Extracting the data from zip files
local_zip = './Images/Validation/validation-horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./Images/Validation')
zip_ref.close()


