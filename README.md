# Road Segmentation Using Deep Learning

## Overview

This project focuses on segmenting roads from aerial images using deep learning techniques. The primary goal is to accurately identify road regions in images to aid in urban planning, autonomous navigation, and mapping.

## Features

- Utilizes the **U-Net** architecture for semantic segmentation.
- Incorporates preprocessing techniques like **Sliding Window** for handling large images.
- Implements data augmentation to improve model generalization.
- Supports training and testing on grayscale and colored images.

## Files

- `train.ipynb`: Notebook for training the model.
- `app.py`: Script for deploying the model and testing predictions.
- `requirements.txt`: Dependencies and libraries required to run the project.
- `trial_4_submission.csv`: Sample submission file with predicted results.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rubarakan/Myproject.git
## Navigate to the project directory:
cd Myproject
## Install the required libraries
pip install -r requirements.txt

Training
To train the model, run the Jupyter Notebook train.ipynb and follow the steps for data preprocessing, model training, and evaluation.

Prediction
To predict road masks for new images:

Run app.py.
Provide the path to the input image(s).
The script outputs the predicted mask.
Results
F1 Score: 0.89
IoU: 0.83
Visual Results
Sample outputs showing input images, ground truth masks, and predicted masks are provided in the results section of the report.

Challenges
Selecting an appropriate model that balances complexity and efficiency.
Handling large images and ensuring accurate segmentation.
Optimizing performance through data augmentation and hyperparameter tuning.
Future Work
Extend the dataset to include diverse geographical regions.
Experiment with advanced architectures like DeepLab and Transformer-based models.
Deploy the model for real-time applications.
Contributors
Ruba Rakan
Karam
Acknowledgments
Special thanks to Dr. Tamam Sarhan for guidance throughout this project

