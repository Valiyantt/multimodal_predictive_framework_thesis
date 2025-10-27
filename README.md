🧠 A Multi-Modal Predictive Framework for Sentiment Analysis

Authors:
Franz Tovie G. Alverio · Lucky Richmon C. Almarinez · Kyla Celine L. Jamilano
Institution: Saint Michael’s College of Laguna – Department of Computer Studies
Advisor: Anna Liza A. Ramos
Year: 2025

---
📖 Overview

This repository contains the source code, datasets, and deployment framework for the thesis “A Multi-Modal Predictive Framework for Sentiment Analysis.”
The project introduces an interpretable deep learning architecture that integrates text, audio, and visual modalities to predict human sentiment more accurately and robustly.

The system leverages the following model components:
- Text: BERT (Bidirectional Encoder Representations from Transformers)
- Audio: Wav2Vec 2.0 + Dual-Branch MLP
- Visual: VGG16 (CNN-based architecture)
- Fusion: Late Fusion Aggregation via Arithmetic Mean

---
🎯 Objectives

- To extract valence-informed emotional features from text, speech, and facial data.
- To build an interpretable model using ensemble-based aggregation functions.
- To enhance performance and robustness against multimodal noise and incomplete inputs.
- To deploy the entire system using Docker for reproducible inference.

---
🧩 Datasets

This study utilizes two publicly available multimodal datasets:

Dataset	Modalities	Emotion Labels	Description
MELD (Multimodal EmotionLines Dataset)	Text, Audio, Visual	7 (Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise)	Derived from the TV series Friends, includes dialogues and aligned audio/video
CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)	Audio, Visual, Text	6 (Anger, Disgust, Fear, Happy, Neutral, Sad)	Contains acted emotional expressions from 91 speakers

---
🧮 Methodology
🔹 Preprocessing

Text: Tokenization, stopword removal, lemmatization, and spelling correction

Visual: Frame extraction, face detection via MTCNN, alignment, resizing, and normalization

Audio: Resampling, duration normalization, pitch shifting, and Gaussian noise augmentation

🔹 Feature Extraction
Modality	Model	Feature Dimension	Description
Text	BERT	768	Semantic and emotional embeddings
Visual	VGG16	4096	Deep facial expression features
Audio	Wav2Vec 2.0 + MFCCs	768 + handcrafted	Combines learned and static prosodic cues
🔹 Fusion & Prediction

Late fusion of modality-specific predictions using an arithmetic mean aggregation function

Balanced weighting prevents modality dominance and enhances interpretability

---
🧠 Model Training & Evaluation
Modality	Model	Optimizer	Key Metric
Text	BERT	AdamW	Macro F1-Score
Visual	VGG16	SGD + Momentum	Accuracy / F1-Score
Audio	Dual-Branch MLP	Adam	ROC-AUC / Accuracy

Robustness Techniques:

Modality-Dropout Augmentation (p = 0.2)

Label smoothing for class imbalance

Early stopping and cross-validation

Benchmark Result:

Achieved 77.25% accuracy on CREMA-D (surpassing prior 70.98% benchmark)


---
⚙️ Deployment

The system is fully containerized with Docker to ensure environment consistency and portability.

Components

Preprocessing pipelines for text, audio, and visual inputs

Trained model weights (BERT, VGG16, MLP)

REST API for multimodal inference

Web dashboard (optional extension)

Deployment Steps
# Clone repository
git clone https://github.com/<your-username>/multimodal-sentiment-framework.git
cd multimodal-sentiment-framework

# Build Docker image
docker build -t multimodal-sentiment .

# Run container
docker run -p 5000:5000 multimodal-sentiment

---
📊 Outputs

Sentiment Polarity: Positive | Neutral | Negative

Emotion Recognition: Anger, Fear, Sad, Happy, Neutral, Disgust, Surprise

Valence-Arousal Analysis: Quantitative emotional scaling

---
🌍 Societal Impact

This framework supports the UN Sustainable Development Goal (SDG) 3: Good Health and Well-Being, contributing to:

Emotion-aware digital platforms

Human-computer interaction

AI-assisted mental health monitoring

---
📚 References

Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, NAACL-HLT, 2019.

Baevski et al., Wav2Vec 2.0: A Framework for Self-Supervised Learning of Speech Representations, NeurIPS, 2020.

Simonyan & Zisserman, VGG16: Very Deep Convolutional Networks for Large-Scale Image Recognition, ICLR, 2015.

Poria et al., MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations, ACL, 2019.

Cao et al., CREMA-D: Crowd-Sourced Emotional Multimodal Actors Dataset, IEEE Trans. Affective Comput., 2014.

---
👥 Contributors

Franz Tovie G. Alverio – Research Lead, Backend Developer, AI Integration

Lucky Richmon C. Almarinez – Data Engineering & Model Training

Kyla Celine L. Jamilano – Data Curation, Documentation, and Testing

Anna Liza A. Ramos – Adviser
