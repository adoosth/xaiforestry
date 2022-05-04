import os
from utils.circles import gen_images
from utils.dataset import build_dataset
from utils.model import build_loaders, build_network, run_train, run_test
from utils.explanables import run_explanable, explanable_RMSE, explanable_cosine, explanable_top1p, explanable_top5p, explanable_top10p
from utils.mask import minimum_enclosing_rectangle

base_folder = '../../../../PlantVillage-Dataset/raw/color'
dataset_dir = './data/experiment_2/datasets/'
class_dirs = ['Apple___healthy', 'Blueberry___healthy',
            'Cherry_(including_sour)___healthy','Grape___healthy',
            'Peach___healthy', 'Potato___healthy', 'Raspberry___healthy',
            'Soybean___healthy', 'Strawberry___healthy', 'Tomato___healthy']
classes = [f.split('_')[0].lower() for f in class_dirs]
                
def build_data():
    os.makedirs(dataset_dir)
    build_dataset([os.path.join(base_folder, dir) for dir in class_dirs],
        dataset_dir, classes, 152, mask_generator = minimum_enclosing_rectangle,
        val_frac = 0.2, test_frac=0.2)

def train_model(model_name = "alexnet", RRR = False):
    num_classes = len(classes)
    model_filename = "./data/experiment_3/" + model_name + '_' + ("RRR" if RRR else "ori") + ".pkl"
    expl_folder_path = "./data/experiment_3/expl_" + ("RRR" if RRR else "ori")
    os.makedirs(expl_folder_path)

    loaders = build_loaders(classes, batchsize = 8)
    net, loaded = build_network(model_filename, num_classes, pretrained=True, pretrained_weights=False)

    run_train(net, loaders, useRRR=RRR, epochs=90, RRR_weight=2, lr=0.00001, filename=model_filename, save_result=True)
    run_test(net, loaders)

    explanable_functions = [explanable_RMSE, explanable_cosine, explanable_top1p, explanable_top5p, explanable_top10p]
    run_explanable(net, loaders, expl_folder_path, explanable_functions, save_results = True)

def main():
    build_data()
    train_model("alexnet", RRR = True) # Set to false for original model

if __name__ == "__main__":
    main()