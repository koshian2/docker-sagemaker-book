mv ./07_s3_inputs/* .
unzip /opt/ml/input/data/training/kagglecatsanddogs_5340.zip -d /opt/ml/input/data/training > /dev/null
rm /opt/ml/input/data/training/kagglecatsanddogs_5340.zip
python 07_train_cats_dogs.py