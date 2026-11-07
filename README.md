# Siemens Software Industry Readiness Program (SIRP) & SmartFall

## About the Program

Starting in August 2026, I participated in the **Siemens Software Industry Readiness Program (SIRP)**, where I gained hands-on experience with machine learning, data engineering, predictive analytics, and low-code application development.

Throughout the program, I worked with **RapidMiner AI Studio, Python, TensorFlow, Keras, and Mendix**, progressing from fundamental machine learning concepts to developing an integrated final-year project prototype.

## Learning Journey

### Week 1 – RapidMiner Fundamentals
- Explored RapidMiner AI Studio and its workflow environment.
- Learned data preprocessing, cleaning, and merging.
- Built initial machine learning models using Decision Trees.
- Practiced cross-validation and model evaluation.

### Week 2 – Machine Learning Techniques
- Worked with regression and classification algorithms.
- Explored clustering techniques.
- Practiced feature engineering and model validation.
- Understood how to evaluate and compare predictive models.

### Week 3 – Data Engineering
- Practiced data merging, pivoting, filtering, and duplicate removal.
- Worked on structured data preparation and transformation.
- Conducted literature surveys on RapidMiner and Mendix.
- Explored the integration of analytics with application development.

---

# Final Project: SmartFall
### AI-Based Adaptive Fall Risk Prediction and Prevention System

## Project Overview

SmartFall is an AI-based fall risk prediction prototype that integrates three machine learning models into a unified **Mendix dashboard**.

The system explores three different prediction tasks:

1. **Stroke Risk Prediction:** Estimates stroke risk using health-related attributes.
2. **Fall Detection:** Identifies fall events using wearable-sensor data.
3. **Gait Change Prediction:** Analyzes gait patterns using sequential data to identify potential changes.

The objective is to demonstrate how machine learning and low-code application development can be combined to build an integrated health-monitoring prototype.

## Machine Learning Models and Results

| Module | Algorithm | Dataset | Accuracy |
|---|---|---|---|
| Stroke Risk | Gradient Boosted Trees | 5,110 Kaggle records | 91.25% |
| Fall Detection | Random Forest | 10,831 UniMiB SHAR sensor windows | 97.9% |
| Gait Change | LSTM | Synthetic data: 24 people, 30 days | 45.8% |

### 1. Stroke Risk Prediction
- Algorithm: Gradient Boosted Trees
- Dataset: 5,110 records from Kaggle
- Accuracy: 91.25%
- Precision: 20.0%
- Recall: 26.5%
- ROC-AUC: 0.833

This model explores stroke risk classification using health-related features.

### 2. Fall Detection
- Algorithm: Random Forest
- Dataset: UniMiB SHAR
- Processed sensor windows: 10,831
- Accuracy: 97.9%
- Precision: 97.1%
- Recall: 95.8%

The model uses wearable-sensor data to distinguish fall events from activities of daily living (ADL).

### 3. Gait Change Prediction
- Algorithm: Long Short-Term Memory (LSTM)
- Dataset: Synthetic sequential data
- Simulated participants: 24
- Duration: 30 days
- Accuracy: 45.8% under 4-fold cross-validation

The LSTM model explores sequential gait measurements and potential changes in movement patterns over time.

## System Integration

The three prediction modules are brought together through a Mendix-based dashboard.

**Technology workflow:**

1. Dataset preparation and preprocessing.
2. Model development and evaluation in RapidMiner and Python.
3. Prediction output preparation.
4. Integration of prediction records into Mendix.
5. Visualization through a unified dashboard.

## Tools and Technologies

- **Machine Learning:** RapidMiner AI Studio
- **Programming:** Python
- **Deep Learning:** TensorFlow, Keras
- **Algorithms:** Random Forest, Gradient Boosted Trees, LSTM
- **Application Development:** Mendix
- **Data Processing:** Data cleaning, transformation, feature engineering, and validation
- **Datasets:** Kaggle Stroke Dataset, UniMiB SHAR, Synthetic Gait Dataset

## Key Learnings and Limitations

This project helped me understand that model accuracy alone does not determine the effectiveness of a machine learning system.

- Precision and recall are especially important when working with imbalanced datasets.
- The fall detection results may be optimistic because overlapping sensor windows can introduce similarities between training and testing samples.
- The gait prediction model uses synthetic data and therefore requires validation on real-world sensor data.
- Stroke prediction has relatively low precision and recall despite its high accuracy, highlighting the importance of evaluating multiple performance metrics.

**Disclaimer:** SmartFall is an academic prototype developed for learning and demonstration purposes. It is not a clinically validated system and must not be used as a medical diagnostic or decision-making tool.

## Program Takeaways

The Siemens SIRP experience strengthened my understanding of the complete machine learning workflow, from data preparation and model development to evaluation and application integration.

It also gave me practical exposure to RapidMiner, Python-based deep learning, data engineering concepts, and Mendix application development.

I am grateful for the opportunity to learn, experiment, and build SmartFall as part of this journey.

---

**Program:** Siemens Software Industry Readiness Program (SIRP)  
**Start:** August 2026  
**Final Project:** SmartFall  
**Project Category:** Machine Learning | Deep Learning | Data Engineering | Low-Code Development
