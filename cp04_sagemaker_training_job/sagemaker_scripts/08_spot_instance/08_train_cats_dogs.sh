mv ./08_spot_instance/* .
unzip /opt/ml/input/data/training/kagglecatsanddogs_5340.zip -d /opt/ml/input/data/training > /dev/null
rm /opt/ml/input/data/training/kagglecatsanddogs_5340.zip
python 08_train_cats_dogs.py