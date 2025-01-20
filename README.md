# Fast and robust visual object recognition in young children

Data, code, and stimuli repository for "Fast and robust visual object recognition in young children"

The repo name KorNet stands for "Kid Object Recognition Networks" -- a play on the CORnet family of models

## Folder Structure

#### analyses
Contains all files for conducting all high-level statistical analyses presented in the main-text  

Human only data figures can be recreated using the sub_analysis.ipynb notebook

Human model comparison figures can be recreated from the model_comparison.ipynb notebook

#### data
Contains all individual participant data files for both children and adults. 

The sub_info.csv file contains subject demographic information for children

#### figures
Contains all figures presented in the main-text and supplemental materials

#### modelling
Contains scripts necessary for extracting model activations and conducting classifcation (i.e., decoding). Below I highlight the files most relevant to the main-text

model_loader.py: A general script for loading model architectures, their transforms, and weights

train.py: The training script used for training VoneNet models

extract_acts.py/extract_acts_layerwise.py: Scripts for extracting feature activations for all training and test stimuli, either from the top layer or from pre-specified layers defined in "all_model_layers.csv"   

decode_image.py/decode_image_layerwise.py: Scripts for predicting the category of a test image after training on naturalistic images using activations from the extract_acts script

train_sizes.xlsx: An excel sheet summarizing the dataset size and total experience of each model as measured by data_set size x eopochs. Information is drawn from papers or model cards describing each model

#### Results
Summary files that are used in subsequent analyses.

group_data folder: thiscontains human only summaries seperated by condition, and summaries containing model performance

model folder: contains performance of each model using each classifier and 




### Naming Conventions
Slightly different naming conventions were used as shorthand for the tasks stimuli and analyses. These do not always line up with the labels used in the manuscript.

#### Tasks 
Complete contour conditions is sometimes referred to as 'Outline' 

Perturbed contour condition is sometimes referred to as 'Pert'

Deleted contour condition is sometimes referred to as 'IC' for illusory contours

### Model Names

Models were often renamed for the main-text to more precisly specify their architecture and training or for readability readible. For a full list of model