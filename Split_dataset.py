from shutil import copyfile
import os
import random

original_folder = 'ML_pile_batch6_3layer_noise'
train_folder = 'train_noise'
val_folder = 'val_noise'
test_folder = 'test_noise'
split_percentage = [0.8, 0.1, 0.1]

data_file_list = os.listdir(original_folder)
random.shuffle(data_file_list)

train_file_list = data_file_list[:int(split_percentage[0]*len(data_file_list))]
val_file_list = data_file_list[int(split_percentage[0]*len(data_file_list)):int(split_percentage[0]*len(data_file_list))+int(split_percentage[1]*len(data_file_list))]
test_file_list = data_file_list[int(split_percentage[0]*len(data_file_list))+int(split_percentage[1]*len(data_file_list)):]
print(len(data_file_list))
print(len(train_file_list))
print(len(val_file_list))
print(len(test_file_list))

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

for file in train_file_list:
    copyfile(original_folder + '/' + file , train_folder + '/' + file)

for file in val_file_list:
    copyfile(original_folder + '/' + file , val_folder + '/' + file)

for file in test_file_list:
    copyfile(original_folder + '/' + file , test_folder + '/' + file)

print('done')