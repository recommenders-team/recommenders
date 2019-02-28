# Example: ExDeepFM
The example below uses the [ml-100k](http://files.grouplens.org/datasets/movielens/ml-100k.zip) data and prepare it into libffm data format for ExDeepFM model training.

## step 1. Download the data and Prepare the data in the required format (Libffm for ExDeepFM)
```
cd data
wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
python ML-100K2Libffm.py
```
Different models require different data format. More detail about data format is described in wiki/*format.md.

## step 2. Edit the configuration file
```
cp example/exDeepFM.yaml config/network_xdeepFM.yaml
```
For detail, see the introduction about the parameter setting in wiki/exDeepFM configuration.md.

## step 3. Train the model
```
python main.py exDeepFM train
```
The first argv element is the directory name for the results. For example, it will create cache/exDeepFM directory to save your cache file, 
checkpoint/exDeepFM to save your trained model, logs/exDeepFM to save your training log.

The second argv element is about the mode. If you want to train a model, you choose "train". If you want to infer results, you choose "infer".

## step 4. Infer the result
Configure which trained model you would like to use for inference in config/network.yaml, and then run:
```
python main.py exDeepFM infer
```