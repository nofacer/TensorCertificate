# TensorCertificate

Help you get prepared for TensorFlow Developer Certificate~

PS：这个不是教程向的，仅用于备考及考试时速查用。详细教程请移步官网。

![](./cover.png)

## Notebooks
* [1_mnist.ipynb](https://colab.research.google.com/drive/1dAbddoKtBydG1ROdZ5nN8cE7z63-4TCn?usp=sharing)<span id="1"></span> 
* [2_flower_classfication.ipynb](https://colab.research.google.com/drive/14er0glheEIKf4M_p_fhP-Vf0qAu8eT9G?usp=sharing)<span id="2"></span> 
* [3_sentiment classification.ipynb](https://colab.research.google.com/drive/1-J1b8tv9vDhKAhdusV3qDijgHqsk6ub3?usp=sharing)<span id="3"></span>
* [4_text_generation.ipynb](https://colab.research.google.com/drive/1XAaIVo9fiZrMYezik9SEzFncPpfrQ-wl?usp=sharing)<span id="4"></span>
* [5_time_series_forecasting.ipynb](https://colab.research.google.com/drive/1GdHiTO7tKSSze_Fw0YttCoP71eZPwk9D?usp=sharing)<span id="5"></span>
## 考点

建议结合notebook对照考点练习。太宽泛的考点如A.1我就直接划掉了。要考试必然要达成的条件如B.1我也就直接划掉了，帮大家减少干扰。
### A. TensorFlow developer skills

1. ~~Know how to program in Python, resolve Python issues, and compile and run Python programs in PyCharm.~~
1. ~~Know how to find information about TensorFlow APIs, including how to find guides and API references on tensorflow.org.~~
1. ~~Know how to debug, investigate, and solve error messages from the TensorFlow API.~~
1. ~~Know how to search beyond tensorflow.org, as and when necessary, to solve your TensorFlow questions.~~
1. ~~Know how to create ML models using TensorFlow where the model size is reasonable for the problem being solved.~~
1. ~~Know how to save ML models and check the model file size.~~
1. ~~Understand the compatibility discrepancies between different versions of TensorFlow.~~

### B. Building and training neural network models using TensorFlow 2.x

1. ~~Use TensorFlow 2.x.~~
1. ~~Build, compile and train machine learning (ML) models using TensorFlow.~~
1. ~~Preprocess data to get it ready for use in a model.~~
1. ~~Use models to predict results.~~
1. ~~Build sequential models with multiple layers.~~
1. ~~Build and train models for binary classification.~~
1. ~~Build and train models for multi-class categorization.~~
1. ~~Plot loss and accuracy of a trained model.~~
1. Identify strategies to prevent overfitting, including augmentation and dropout. [1](#1) [2](#2)
1. Use pretrained models (transfer learning). [2](#2)
1. Extract features from pre-trained models. [2](#2)
1. ~~Ensure that inputs to a model are in the correct shape.~~
1. ~~Ensure that you can match test data to the input shape of a neural network.~~
1. ~~Ensure you can match output data of a neural network to specified input shape for test data.~~
1. ~~Understand batch loading of data.~~
1. Use callbacks to trigger the end of training cycles. [4](#4)
1. Use datasets from different sources.
1. Use datasets in different formats, including json and csv.
1. Use datasets from tf.data.datasets.

### C. Image classification

1. Define Convolutional neural networks with Conv2D and pooling layers. [2](#2)
1. Build and train models to process real-world image datasets. [2](#2)
1. Understand how to use convolutions to improve your neural network. [2](#2)
1. Use real-world images in different shapes and sizes. [2](#2)
1. Use image augmentation to prevent overfitting. [2](#2)
1. Use ImageDataGenerator. [2](#2)
1. Understand how ImageDataGenerator labels images based on the directory structure. [2](#2)

### D. Natural language processing (NLP)

1. ~~Build natural language processing systems using TensorFlow.~~
1. Prepare text to use in TensorFlow models. [3](#3)
1. Build models that identify the category of a piece of text using binary categorization [3](#3)
1. Build models that identify the category of a piece of text using multi-class categorization [4](#4)
1. Use word embeddings in your TensorFlow model. [3](#3)
1. Use LSTMs in your model to classify text for either binary or multi-class categorization. [3](#3)
1. Add RNN and GRU layers to your model. [4](#4)
1. Use RNNS, LSTMs, GRUs and CNNs in models that work with text. [3](#3) [4](#4)
1. Train LSTMs on existing text to generate text (such as songs and poetry) [4](#4)

### E. Time series, sequences and predictions

1. Train, tune and use time series, sequence and prediction models. [5](#5)
1. Prepare data for time series learning. [5](#5)
1. Understand Mean Average Error (MAE) and how it can be used to evaluate accuracy of sequence models. [5](#5)
1. Use RNNs and CNNs for time series, sequence and forecasting models. [5](#5)
1. Identify when to use trailing versus centred windows. 
    ```
    center_ma(t) = mean(obs(t-1), obs(t), obs(t+1))
    A center moving average can be used as a general method to remove trend and seasonal components from a time series, a method that we often cannot use when forecasting.

    trail_ma(t) = mean(obs(t-2), obs(t-1), obs(t))
    Trailing moving average only uses historical observations and is used on time series forecasting.
    ```
1. Use TensorFlow for forecasting. [5](#5)
1. Prepare features and labels. [5](#5)
1. Identify and compensate for sequence bias.
(When respondents tend to favor objects because of their position in a list or sequence. The objects at the beginning and at the end of a list can be remembered more than those occurring in the middle. Usual practice is to rotate a list to eliminate this type of bias.) 
1. Adjust the learning rate dynamically in time series, sequence and prediction models. [5](#5)