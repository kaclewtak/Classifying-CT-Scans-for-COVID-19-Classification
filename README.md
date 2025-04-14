# Project Overview
Modern medical imaging has proven time and again to be a field which could see immense benefit from the utilization of computer vision networks in aiding professionals of the medical field. COVID-19 continues to affect individuals with compromised immune systems and can often lead to secondary infections. For this project, we look into the image classification abilities of several model architectures: ResNetV2, EfficientNetV2, and ConvNeXt.

The goal of this project is to test how well these architectures are able to perform in a classification task, as well as conclude whether or not they might be suitable for a real world implementation in a professional medical setting.

A high level overview of what steps were taken throughout the notebook. 
![image](https://github.com/user-attachments/assets/d5822313-b8de-4574-9155-004c5fb759b5)

# Description of Methods

Data Ingestion and Exploration:
The notebook begins by loading CT scan images from three categories (negative, positive, and non-informative) stored in distinct folders. An exploratory data analysis (EDA) is performed to inspect image dimensions, distributions, and class imbalances.

Image Pre-Processing:
Images are first resized uniformly to 512×512 pixels. Since most images contain three channels with redundant information, the extra channels are removed to produce a proper grayscale image. A feature extraction process is then applied using local thresholding and morphological operations (including erosion, closing, and edge detection) to isolate lung features.

Data Augmentation:
To mitigate issues with class imbalance, an augmentation pipeline using TensorFlow’s ImageDataGenerator performs random rotations and translations. This generates additional images for underrepresented classes and helps create robust training datasets.

Model Architecture and Training:
Several convolutional neural network architectures are implemented (EfficientNetV2, ConvNeXt, and ResNetV2 variants). A custom top (consisting of global average pooling, batch normalization, dropout, and a dense output layer with softmax) is appended to each base model. The models are trained using a combination of early stopping and checkpoint callbacks, and evaluated on a separate test dataset.

Evaluation and Visualization:
The pipeline includes comprehensive evaluation metrics such as confusion matrices, classification reports, and ROC curves to assess model performance, providing insights into accuracy, precision, recall, and overall model robustness.

# Conclusions

Ultimately, I conclude that of the model architectures tested, EfficientNetV2 shows the most promise for this type of medical dataset. All models belonging to this architecture showed a remarkable ability to successfully classify the various image classes. Such classification models show promise as tools for medical professionals. This would aid them with their work, reducing the time to diagnosis, and potentially increasing their ability provide care where necessary.

Some limitations of this project include compute resources. While I originally intended to train all models locally on my machine, I encountered VRAM issues which forced me to utilize my backup strategy of training on Google Colab. Another limitation is domain specific knowledge. This project was performed with only cursory knowledge of the medical field. Subsequent research had to be performed in order to expand understanding of the materials used in this notebook.

Future works including this data would include the training existing EfficientNetV2 models, as well as larger EfficientNetV2 models, for significantly more epochs and testing to see if proper convergence can be reached. Likewise, it would be of academic interest to see if the model could then be used for transfer learning and applied to other medical images which require classification tasks performed.
