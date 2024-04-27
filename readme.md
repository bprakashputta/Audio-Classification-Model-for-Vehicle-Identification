<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://i.imgur.com/6wj0hh6.jpg" alt="Project logo"></a>
</p>

<h3 align="center">Audio Classification Model for Vehicle Identification</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> Few lines describing your project.
    <br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Usage](#usage)
- [Built Using](#built_using)
- [TODO](../TODO.md)
- [Contributing](../CONTRIBUTING.md)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>

The aim of this project is to develop an advanced model for classifying vehicle types using audio files as input data. The model's primary objective is to accurately predict whether a given audio recording represents the sound of a car or a truck. Leveraging state-of-the-art machine learning techniques, the model will analyze the audio features extracted from the files to make these predictions.

## Requirements <a name = "requirements">

- **Python 3.6+**
- **NumPy (`pip install numpy`)**
- **Pandas (`pip install pandas`)**
- **Scikit-learn (`pip install scikit-learn`)**
- **SciPy (`pip install scipy`)**
- **librosa (`pip install librosa`)**
- **MatplotLib (`pip install matplotlib`)**
- **Seaborn (`pip install seaborn`)**
- **WTForms (`pip install wtforms`)**
- **Tensorflow (`pip install tensorflow>=1.15`)**
- **Keras (`pip install keras`)**
- **SVC (`pip install svm`)**

## Dataset <a name = "dataset">

### Dataset Description

The dataset comprises a collection of audio recordings capturing various vehicle sounds, accompanied by corresponding metadata stored in JSON format. Each entry in the dataset includes details such as the camera identifier (cam), classification probability (probs), class label indicating whether the sound belongs to a car or a truck (cls), timestamp (dto), and additional attributes.

### Metadata Attributes:

Camera Identifier (cam): Indicates the camera source of the audio recording.

Classification Probability (probs): Represents the probability of the classification result.

Class Label (cls): Specifies whether the sound corresponds to a car (class 2) or a truck (class 7).

Timestamp (dto): Denotes the date and time of the audio recording.

Other Fields: Additional attributes include intersection coordinates (intersection), bounding box coordinates (box), frame timestamp (frame_dto), sequence length (seq_len), and file paths for full and debug images.

Audio Files
The dataset includes audio files corresponding to each entry, stored in the specified paths under the snd field. These audio files capture the auditory signatures of vehicles, enabling the classification model to learn and distinguish between car and truck sounds.

## Usage <a name="usage">

### Training and Evaluation:

The dataset serves as a valuable resource for training and evaluating machine learning models for vehicle classification based on audio data.

### Research and Development:

Researchers and developers can leverage the dataset to explore novel approaches for vehicle sound analysis and classification.

### Real-world Applications:

The insights gained from this dataset can contribute to various applications, including traffic monitoring, urban planning, and surveillance systems.

## Model Architecture:

For this project, two different machine learning architectures were employed: Support Vector Machines (SVM) and Keras Sequential models. Both approaches were utilized to train and evaluate the dataset for vehicle classification based on audio features.

### SVM (Support Vector Machines):

Support Vector Machines are powerful supervised learning algorithms used for classification tasks. In this project, SVM was employed due to its ability to handle high-dimensional data effectively. SVM works by finding the hyperplane that best separates the data into different classes while maximizing the margin between the classes.

#### Feature Extraction:

Relevant audio features were extracted from the dataset to represent the characteristics of car and truck sounds.
X_train shape: (2479, 13, 100)
X_test shape: (620, 13, 100)
y_train shape: (2479,)
y_test shape: (620,)

#### Model Training:

The SVM classifier was trained using the extracted audio features to learn the decision boundary between car and truck classes.

#### Model Evaluation:

The performance of the SVM model was evaluated using metrics such as accuracy, precision, recall, and F1-score to assess its effectiveness in classifying vehicle types.

### Keras Sequential Models:

Keras Sequential models offer a flexible and intuitive way to build neural network architectures. In this project, Sequential models were utilized to explore deep learning-based approaches for vehicle classification from audio data.

#### Feature Learning:

The Sequential model was designed to learn hierarchical representations of audio features through multiple layers of neural networks.

#### Model Training:

The Sequential model was trained on the dataset using optimization techniques such as gradient descent to minimize classification errors.

#### Model Evaluation:

Similar to SVM, the performance of the Sequential model was evaluated using standard classification metrics to measure its accuracy and generalization ability.

By employing both SVM and Keras Sequential models, this project aims to compare the effectiveness of traditional machine learning approaches with deep learning techniques for vehicle classification based on audio features. This hybrid approach allows for a comprehensive evaluation of model performance and provides insights into the most suitable methods for this specific task.

## Usage:

The trained model can be deployed for practical applications requiring real-time vehicle classification. Users can input audio samples into the model, and it will output predictions indicating whether the sound corresponds to a car or a truck.

## Repository Structure:

Data: Contains the audio files and their corresponding class labels.
Model: Includes the code for training, evaluating, and deploying the classification model.
Documentation: Contains detailed instructions on how to use the model, along with any additional information required.
Dependencies:
The project relies on various libraries and frameworks, including but not limited to:

Python
TensorFlow or PyTorch (for deep learning)
NumPy
SciPy
Librosa (for audio feature extraction)
Scikit-learn (for model evaluation)
Contribution Guidelines:
Contributions to the project, such as code improvements, bug fixes, or feature enhancements, are welcome. Please refer to the contribution guidelines in the repository for more information.

## ✍️ Authors <a name = "authors"></a>

- [@kylelobo](https://github.com/kylelobo) - Idea & Initial work

See also the list of [contributors](https://github.com/kylelobo/The-Documentation-Compendium/contributors) who participated in this project.

## 🎉 Acknowledgements <a name = "acknowledgement"></a>

- Hat tip to anyone whose code was used
- Inspiration
- References
