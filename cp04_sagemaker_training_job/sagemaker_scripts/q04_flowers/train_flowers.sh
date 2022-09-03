mv ./q04_flowers/* .
tar -zxvf /opt/ml/input/data/training/flower_photos.tgz -C /opt/ml/input/data/training > /dev/null
rm /opt/ml/input/data/training/flower_photos.tgz
python train_flowers.py