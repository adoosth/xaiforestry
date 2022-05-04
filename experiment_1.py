import os
from utils.circles import gen_images
from utils.dataset import build_dataset
from utils.model import build_loaders, build_network, run_train, run_test
from utils.explanables import run_explanable, explanable_RMSE, explanable_cosine, explanable_top1p, explanable_top5p, explanable_top10p

healthy_dir = './data/PlantVillage-Dataset/raw/color/Pepper,_bell___healthy'
fake_dir = './data/experiment_1/fake/'
dataset_dir = './data/experiment_1/datasets/'
classes = ["healthy", "fake"]

def build_data():
    os.makedirs(dataset_dir)
    os.makedirs(fake_dir)
    gen_images(healthy_dir, fake_dir, os.listdir(healthy_dir), radiusRange = (0, 20), num_circles=3, copies=1, opacity=0.3, save_mask=True)
    build_dataset([healthy_dir, fake_dir], dataset_dir, classes, classes, random_state=42)

def train_model(model_name = "alexnet", RRR = False):
    num_classes = len(classes)
    model_filename = "./data/experiment_1/" + model_name + '_' + ("RRR" if RRR else "ori") + ".pkl"
    expl_folder_path = "./data/experiment_1/expl_" + ("RRR" if RRR else "ori")
    os.makedirs(expl_folder_path)

    loaders = build_loaders(classes, batchsize = 8)
    net, loaded = build_network(model_filename, num_classes, pretrained=True, pretrained_weights=False)

    run_train(net, loaders, useRRR=RRR, epochs=90, RRR_weight=1.5,  lr=0.000002, filename=model_filename, save_result=True)
    run_test(net, loaders)

    explanable_functions = [explanable_RMSE, explanable_cosine, explanable_top1p, explanable_top5p, explanable_top10p]
    run_explanable(net, loaders, expl_folder_path, explanable_functions, save_results = True)

def main():
    build_data()
    train_model("alexnet", RRR = True) # Set to false for original model

if __name__ == "__main__":
    main()