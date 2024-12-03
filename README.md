# Hybrid Crop Recommendation Model
This repository contains a hybrid machine learning and deep learning model designed for crop recommendation. The model integrates Naive Bayes, Decision Tree, and Deep Neural Networks to deliver robust predictions for recommending the most suitable crop based on given environmental parameters.

# Overview
This project addresses the need for efficient crop recommendations tailored to specific soil, weather, and environmental conditions. By combining traditional machine learning techniques with advanced deep learning, the model offers:
a)Probabilistic insights using Naive Bayes.
b)Improved classification using a Decision Tree.
c)Pattern recognition and robust predictions using a Deep Neural Network.

# Features
1)Hybrid integration of machine learning and deep learning.
2)Multi-class classification for various crops.
3)Early stopping mechanism to prevent overfitting.
4)Visualization of metrics like confusion matrices, ROC curves, and training performance.

# Technologies Used
a)Programming Language: Python
b)Libraries:
1)Machine Learning: scikit-learn
2)Deep Learning: TensorFlow/Keras
3)Data Manipulation: Pandas, NumPy
4)Visualization: Matplotlib, Seaborn

# Data Description
Dataset: Crop Recommendation Dataset
1)Columns include environmental parameters like soil type, temperature, humidity, pH level, and rainfall.
2)Target variable: Crop label.

Preprocessing:
1)Target labels were encoded using LabelEncoder.
2)Features were standardized using StandardScaler.

# Model Architecture
Step 1: Naive Bayes
Predicts probabilities based on raw feature inputs.
Step 2: Decision Tree
Utilizes probabilistic outputs of Naive Bayes for classification.
Step 3: Deep Learning
A fully connected neural network with hidden layers, dropout for regularization, and softmax for multi-class outputs.

# Results
The hybrid model demonstrates excellent performance, achieving high accuracy and consistent cross-validation scores. The results indicate that the combination of machine learning and deep learning effectively handles the complexities of crop recommendation.
# Training and Validation Metrics
Training Accuracy: Progressed from 7.15% (Epoch 1) to 85.78% (Epoch 23).
Validation Accuracy: Reached a peak of 99.09%.
Validation Loss: Stabilized around 0.082 after Epoch 16, showcasing minimal overfitting.
# Test Set Evaluation
Accuracy: 99.09%
Precision: 99.11%
Recall: 99.09%
F1-Score: 99.09%
# Cross-Validation Results (Mean ± Standard Deviation)
Accuracy: 99.32% ± 0.25%
Precision: 99.33% ± 0.25%
Recall: 99.32% ± 0.25%
F1-Score: 99.32% ± 0.25%
# Insights from Metrics
1)Model Performance:
The model performs exceptionally well across all metrics, indicating balanced predictions with minimal misclassification.
2)Consistency:
Cross-validation results have low variance, signifying robust and reliable performance on unseen data.
3)Efficiency:
Validation accuracy plateaued early, suggesting efficient training and convergence.

# Future Visualizations
1)To further validate the model, the following visualizations can be added:
Confusion Matrix: To understand misclassifications.
ROC Curves: To analyze the true positive and false positive rates for each crop class.
Loss and Accuracy Graphs: To showcase training progression.
2)Integration of Advanced Features:
Include weather forecasts and real-time data inputs.
Use satellite imagery for additional insights into soil health.
3)Model Improvements:
Experiment with ensemble models or transformers.
Fine-tune hyperparameters using grid search or Bayesian optimization.
4)Deployment:
Develop a web-based or mobile application for farmers.
Integrate IoT devices for real-time recommendations.
5)Multi-Language Support:
Enable the application to support regional languages for better accessibility.

# Acknowledgments
The dataset was sourced from Kaggle.
Thanks to my mentors and colleagues for their guidance.

