# Lightweight CNN Model with Attention Mechanism for Heart Sound Signal Classification
##### 🩺 Project Overview

This project presents a **lightweight Convolutional Neural Network (CNN) model with an attention mechanism** designed to classify heart sound signals efficiently. The goal is to detect abnormalities in heartbeats, which can help in the **early diagnosis of heart diseases.**

The model is built to be simple, fast, and lightweight—making it suitable even for **low-resource environments** such as mobile devices or embedded health-monitoring systems.

##### 🔍 Motivation

Heart disease is a leading cause of death worldwide. Early detection through **heart sound analysis (phonocardiograms or PCG signals)** can save lives. However, many existing deep learning models are heavy and require a lot of computational power.

This project solves that by building a **lightweight yet accurate model**, combining:

- CNNs (to extract key features from the heart sound signals)

- Attention mechanisms (to focus on important parts of the signal)

##### 🎯 Objectives

  - Build a lightweight CNN model for PCG classification

  - Improve accuracy by adding an attention mechanism

  - Maintain a small model size and low training time

  - Achieve good results on heart sound classification datasets

##### 🧠 What is a Heart Sound Signal?

A heart sound signal is the **audio recording of the heartbeat**, usually captured using a digital stethoscope. It includes sounds like **lub-dub**, murmurs, and other signals. Analyzing this signal helps detect **cardiac disorders** like valve defects or arrhythmias.
##### 🧪 Dataset Used

We used a heart sound dataset that contains:

  - **Normal** heart sounds

  - **Abnormal** heart sounds (with murmurs, extra sounds, etc.)

![Reference Image](/figure1.png)

*Figure 1. A PCG (center tracing), with simultaneously recorded ECG (lower tracing) and the four states of the PCG recording; S1, Systole, S2 and Diastole.*

Each audio clip is preprocessed and converted to a Log-Mel spectrogram, which represents both time and frequency information.
##### 🛠️ Preprocessing Steps

1. **Load WAV audio files**

2. **Apply preprocessing:**
   
   - Normalization
   - Silence removal (if needed)

3. **Convert audio to Log-Mel spectrograms** using `librosa`

4. **Split dataset into:**

    - 80% Training data

    - 20% Testing data

##### 🧱 Model Architecture

The model contains:

1. **CNN Layers**
    - Extract spatial and frequency-based features from spectrograms

2. **Batch Normalization and Dropout**
    - For regularization and faster convergence

3. **Attention Mechanism**
    - Learns which parts of the heart sound are most important

4. **Fully Connected (Dense) Layers**
    - For final classification (Normal / Abnormal)
  

`Input (Log-Mel Spectrogram)`
    ↓
`Conv2D → BatchNorm → ReLU → Dropout`
    ↓
`Conv2D → BatchNorm → ReLU → Dropout`
    ↓
`Attention Mechanism`
    ↓
`Flatten → Dense → Output Layer (Softmax)`

##### 🧠 Attention Mechanism (Why?)

Attention helps the model **focus only on relevant regions** of the spectrogram—like where a murmur might appear—making the model more **interpretable and accurate.**
##### 🧪 Evaluation Metrics

We used:

   - **Accuracy**

   - **Precision**

   - **Recall**

   - **F1-score**

   - **Confusion Matrix** (to visualize correct and incorrect predictions)

    

##### ✅ Results
```bash
Metric	                           Value
Training Accuracy	           ~90%
Testing Accuracy	           ~85-88%
Model Size	                   Lightweight (few MBs)
Training Time	                   Very Fast
```

###### ✅ The model achieved **high accuracy with low memory usage** and is easy to deploy.
##### 🧠 Key Features

   - 🪶 **Lightweight** – Suitable for real-time systems

   - 🔎 **Attention Layer** – Focuses on key parts of the audio signal

   - 💾 **Small Model Size** – Easy to deploy on mobile or embedded devices

   - 🏥 **Real-world Application** – Can be used in healthcare devices

##### 📁 Project Structure
```bash
.
├── data/                         # Audio files (WAV format)
├── preprocessing.py             # Convert audio to spectrogram
├── model.py                     # CNN + Attention model code
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── utils/                       # Utility functions
├── requirements.txt             # Required Python libraries
├── README.md                    # Project documentation
└── notebook.ipynb               # Colab/Jupyter notebook (full pipeline)
```

##### ▶️ How to Run
###### 🧰 Requirements

Install the required libraries:

`pip install -r requirements.txt`

###### 🏃 Run Training

`python train.py`

###### 🔍 Evaluate Model

`python evaluate.py`

##### 🔗 Google Colab

Run the full project interactively in Colab:
📎[![Open in Google Colab](https://img.shields.io/badge/Open%20in-Google%20Colab-orange?logo=googlecolab)](https://colab.research.google.com/drive/1qubvzeBC19viB5uk0fKrPbhPktJbrYbC)


##### 📦 GitHub Repo

Visit the complete code and documentation here:
🔗 [GitHub Repository](https://github.com/Somenpradhan/ECG-Signal)

##### 📚 References

1. PhysioNet Challenge: Heart Sound Datasets

2. CNNs and Attention Mechanisms in Deep Learning

3. GitHub - ECG Signal

4. Colab - Heart Sound Classification Project

##### 🔤 List of Abbreviations
```brash
Term    	Full Form

CNN	Convolutional Neural Network
PCG	Phonocardiogram
AI	Artificial Intelligence
ACF	Auto-Correlation Function
```
##### 🙌 Acknowledgments

- Inspired by research in **AI for healthcare**

- Special thanks to the open-source community

- Developed as part of a deep learning project on biomedical signals

##### 📬 Contact

Feel free to connect or ask questions:
📧 Email: somenpradhan135@gmail.com

🔗 GitHub: @Somenpradhan