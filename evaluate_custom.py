#!/usr/bin/env python3

from config_class import Config
from evaluate_class import LoadModel
from subroutines.HDF5 import HDF5DatasetGeneratorMask
from subroutines.visualize_predictions import Visualize_Predictions

folder = {}
# Use this to easily run the code in different directories/devices
folder["initial"] = "/home/robot/extars/"
# The path where the repository is stored
folder["main"] = folder["initial"] + "crack_detection_CNN_masonry/"
folder["images"] = folder["main"] + "dataset/custom_images/"
folder["masks"] = folder["main"] + "dataset/custom_masks/"
outputPath = folder["main"] + "output/hdf5/custom/test.hdf5"
IMAGE_DIMS = (224, 224, 3)
info = "crack_detection"
# Batch size
BS = 4
# load pretrained model/weights
args = {}
args["main"] = folder["main"]
args["output"] = args["main"] + "output/"
args["model"] = "Unet"  # 'Deeplabv3', 'Unet', 'DeepCrack', 'sm_Unet_mobilenet'
args["counter"] = 19053
args["model_json_folder"] = args["output"] + "model_json/"
args["model_json"] = (
    args["model_json_folder"] + info + "_{}.json".format(args["counter"])
)
args["weights"] = args["output"] + "weights/"
args["pretrained_filename"] = "crack_detection_23985_epoch_81_F1_score_dil_0.812.h5"
args["binarize"] = True
args["predictions"] = args["output"] + "predictions/"
args["save_model_weights"] = "weights"
args["EVAL_HDF5"] = args["output"] + "hdf5/custom/test.hdf5"
args["predictions_subfolder"] = '{}{}/'.format(args['predictions'], args['pretrained_filename'])
args['predictions_dilate'] = True

cnf = Config("")  # Dummy

# Start procedure.
model = LoadModel(args, IMAGE_DIMS, BS).load_pretrained_model()

# Do not use data augmentation when evaluating model: aug=None
evalGen = HDF5DatasetGeneratorMask(
    args["EVAL_HDF5"], BS, aug=None, shuffle=False, binarize=args["binarize"]
)

# Use the pretrained model to fenerate predictions for the input samples from a data generator
predictions = model.predict_generator(
    evalGen.generator(),
    steps=evalGen.numImages // BS + 1,
    max_queue_size=BS * 2,
    verbose=1,
)

# Define folder where predictions will be stored
predictions_folder = "{}{}/".format(args["predictions"], args["pretrained_filename"])
# Create folder where predictions will be stored
cnf.check_folder_exists(predictions_folder)

# Visualize  predictions
# Create a plot with original image, ground truth and prediction
# Show the metrics for the prediction
# Output will be stored in a subfolder of the predictions folder (args['predictions_subfolder'])
Visualize_Predictions(args, predictions)
