import tarfile
import os
import random
from sklearn.model_selection import train_test_split
os.chdir("d:/GWDG/gitlab/adoosth/DataGeneration")

def build_dataset(src_dirs, dst_dir, names, masked_names, val_frac = 0.2, test_frac = 0.2, random_state = 42):
    src_dirs = [os.path.join(i, "") for i in src_dirs]
    dst_dir = os.path.join(dst_dir, "")
    traintar = tarfile.open(dst_dir + "traindata.tar", "w")
    valtar = tarfile.open(dst_dir + "validationdata.tar", "w")
    testtar = tarfile.open(dst_dir + "testdata.tar", "w")
    all_files = []
    file_dirs = []
    for i in range(len(src_dirs)):
        src_dir = src_dirs[i]
        name = names[i]
        filenames = os.listdir(src_dir)
        if name in masked_names:
            filenames = [f for f in filenames if f.split('.')[0] != "mask"]
        all_files += filenames
        file_dirs.append(filenames)
    for i in range(len(src_dirs)):
        src_dir = src_dirs[i]
        name = names[i]
        filenames = file_dirs[i]
        train_names, val_names = train_test_split(filenames, test_size=val_frac, random_state=random_state)
        train_names, test_names = train_test_split(train_names, test_size=test_frac/(1.0-val_frac), random_state=random_state)
        for f in train_names:
            traintar.add(src_dir + f, arcname=name + "_" + str(all_files.index(f)) + '.' + f.split('.')[-1])
            if name in masked_names:
                traintar.add(src_dir + "mask." + f, arcname="mask" + "_" + str(all_files.index(f))+ '.' + f.split('.')[-1])
        for f in val_names:
            valtar.add(src_dir + f, arcname=name + "_" + str(all_files.index(f))+ '.' + f.split('.')[-1])
            if name in masked_names:
                valtar.add(src_dir + "mask." + f, arcname="mask" + "_" + str(all_files.index(f))+ '.' + f.split('.')[-1])
        for f in test_names:
            testtar.add(src_dir + f, arcname=name + "_" + str(all_files.index(f))+ '.' + f.split('.')[-1])
            if name in masked_names:
                testtar.add(src_dir + "mask." + f, arcname="mask" + "_" + str(all_files.index(f))+ '.' + f.split('.')[-1])
    traintar.close()
    valtar.close()
    testtar.close()