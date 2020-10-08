from google_drive_downloader import GoogleDriveDownloader as gdd
import zipfile
import os
import glob


file_id = '1kySNos1iSDQZUjqJZPV69cDGEYFaKhM1'
dest_path = './data/data-science-bowl-2018.zip'


gdd.download_file_from_google_drive(file_id=file_id, dest_path=dest_path, unzip=True)


with zipfile.ZipFile('data/stage1_test.zip', 'r') as z:
  z.extractall('data/test/')

with zipfile.ZipFile('data/stage1_train.zip', 'r') as z:
  z.extractall('data/train/')


deprecated_data = glob.glob('data/*.zip')


for file in deprecated_data:
  os.remove(file)
