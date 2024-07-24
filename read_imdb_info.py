import h5py
import numpy as np
# 打开HDF5文件
import json

json_path = 'mmimdb/split.json'
with open(json_path, 'r') as f:
    a = json.load(f)
    import pdb
    pdb.set_trace()

# with h5py.File('multimodal_imdb.hdf5', 'r') as file:
#     # 查看文件中包含的数据集名称
#     print("Datasets in HDF5 file:", list(file.keys()))
#     #Datasets in HDF5 file: ['features', 'genres', 'images', 'imdb_ids', 'sequences', 'three_grams', 'vgg_features', 'word_grams']
#     # 读取数据集中的数据
#     feature = file['features']  # 将'dataset_name'替换为您要读取的数据集名称
#     print("Shape of the feature:", feature.shape)
#     genras = file['genres']
#     print("Shape of the genras:", genras.shape)
#     images = file['images']
#     print("Shape of the images:", images.shape)
#     imdb_ids = file['imdb_ids']
#     print("Shape of the imdb_ids:", imdb_ids.shape)
#     sequences = file['sequences']
#     print("Shape of the sequences:", sequences.shape)
#     three_grams = file['three_grams']
#     print("Shape of the three_grams:", three_grams.shape)
#     vgg_features = file['vgg_features']
#     print("Shape of the vgg_features:", vgg_features.shape)
#     word_grams = file['word_grams']
#     print("Shape of the word_grams:", word_grams.shape)
#     import pdb;     pdb.set_trace()