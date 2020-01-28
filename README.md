# Intel Image Classification

Intel Image Classification: see [here](https://www.kaggle.com/puneet6060/intel-image-classification).
You can get the data from this link. 

## Required libraries
- Tensorflow
- Keras
- scikit-learn
- Seaborn
- Pandas
- Matplotlib
- OpenCV for Python
- tf-nightly, if you want to use tensorboard in Jupyter Notebook (see [here](https://www.dlology.com/blog/how-to-run-tensorboard-in-jupyter-notebook/))

This notebook was designed to be run on Google Colab. For this to work,
you need the dataset on your Google Drive. 
If you have the dataset on your local computer and a 
machine learning capable GPU, you can change the variables train_directory
and test_directory to the corresponding directories and then execute the notebook.
The data record should be in the preconstructed folder structure. 
The validation set is a subset (default: 25 percent) of the
train set.

## Approaches to classify images:
- Transfer Learning: from pretrained VGG16-Model
- Image Augmentation: Random crops, blurring, left-right-flips
- Regularization: Dropouts, BatchNormalization

## Results: 
After Training for 6 Epochs
- Accuracy on train set: 0.9281
- Accuracy on validation set: 0.9229
- Accuracy on test set: 0.914

![ConfusionMatrix](/presentation/confusion_matrix.JPG?raw=true)


