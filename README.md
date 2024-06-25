# Hair Segmentation 

## Architecture

The hair segmentation model consists of two stages: a hair region extractor and a border refinement model. The model architecture utilizes MobileNetV2 as the backbone network and an ASPP (Atrous Spatial Pyramid Pooling) module for faster and more accurate predictions.

The hair region extractor is responsible for extracting the initial hair regions from the input images. It leverages the MobileNetV2 architecture, which is a lightweight and efficient convolutional neural network. MobileNetV2 is known for its ability to balance between accuracy and computational efficiency.

The border refinement model further refines the hair segmentations by focusing on the boundaries of the hair regions. It uses the custonm module, which employs atrous convolutions at multiple scales to capture contextual information effectively. This helps in improving the accuracy of the hair segmentations, especially around the hair edges.

The models were trained on a combination of the k-hair dataset and a company-specific dataset. The k-hair dataset is a widely used benchmark dataset for hair segmentation tasks, while the company-specific dataset contains additional data specific to the company's requirements. Training on a diverse dataset helps the models to generalize well and perform accurately on various hair types and styles.

## Setup Instructions
To run the hair segmentation model, follow these steps:

### Windows
1. Open a command prompt or PowerShell window.
2. Navigate to the project directory: `cd C:\path\to\hair-segment`
3. Create a virtual environment: `python -m venv venv`
4. Activate the virtual environment: `venv\Scripts\activate`
5. Install the required dependencies: `pip install -r requirements.txt`
6. Run the main.py script: `python main.py`

### Linux
1. Open a terminal.
2. Navigate to the project directory: `cd /home/aditya/Documents/projects/hair-segment`
3. Create a virtual environment: `python3 -m venv venv`
4. Activate the virtual environment: `source venv/bin/activate`
5. Install the required dependencies: `pip install -r requirements.txt`
6. Run the main.py script: `python main.py`

Make sure you have Python installed on your system before following these steps.

![alt text](blob/first.png?raw=true)
![alt text](blob/second.png?raw=true)
![alt text](blob/third.png?raw=true)
![alt text](blob/fourth.png?raw=true)
![alt text](blob/fifth.png?raw=true)

