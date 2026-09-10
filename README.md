# MLPClothing

# Overview
This project develops a machine learning model to predict clothing sizes (S, M, L) based on individual body measurements (weight, age, height), while also accounting for brand-specific sizing variations. It utilizes a Multi-Layer Perceptron (MLP) as a backbone model, calibrated with brand-specific weight adjustments derived from real-world size charts.

# Features
Brand Calibration: Adjusts predictions based on how different brands (e.g., H&M, Uniqlo, GAP) typically run (small, true to size, large).
Feature Engineering: Incorporates Body Mass Index (BMI) and Weight-to-Height (W/H) ratio as additional features for improved prediction accuracy.
MLP Backbone: A neural network (MLPClassifier) trained on a synthetic 'true-to-size' dataset (Nike) serves as the core prediction engine.
Comprehensive Evaluation: Includes detailed classification reports, confusion matrices, and cross-validation to assess model performance.
Insightful Visualizations: Generates plots for MLP training loss, network architecture, brand accuracy comparison, decision boundaries, and a brand size grid.
Interactive Predictor: An easy-to-use interface to input body measurements and receive personalized size recommendations across multiple brands.

# How It Works
Data Preparation: The project uses final_test.csv (Dataset A) for model development due to its consistent feature correlation and size distribution. A synthetic dataset is also generated using real brand size charts to create a 'true-to-size' baseline (Nike) and simulate various brand sizings.
MLP Backbone Training: A Multi-Layer Perceptron is trained on the synthetic Nike dataset, which represents a 'true-to-size' reference. This MLP learns the fundamental relationship between body measurements and clothing sizes.
Brand Calibration: For each brand, a 'weight shift' value is calculated. This value represents how much a person's perceived weight needs to be adjusted for a specific brand to match the 'true-to-size' MLP's expectations. For example, a brand that 'runs small' will have a positive weight shift, effectively making the person seem 'heavier' to the model, leading to a larger size prediction.
Prediction: When a user inputs their measurements, the model first applies the appropriate weight shift for the selected brand, then calculates engineered features (BMI, W/H ratio), and finally feeds this adjusted data to the pre-trained MLP backbone to get a size prediction.

# Datasets
final_test.csv: Used for analysis and comparison, specifically Dataset A, which showed reasonable correlations between body measurements and clothing size categories.
personalized_clothing_dataset.csv: Dataset B was considered but excluded due to unreasonable average means and lack of clear correlation between measurements and size, indicating potential noise.
Synthetic Brand Data: Generated based on BRAND_CHARTS and BMI_RANGES to create controlled training and evaluation sets for the MLP backbone and brand calibration.
Model Performance
The MLP backbone achieved an accuracy of approximately 95.5% on the test set. After brand calibration, individual brand prediction accuracies and macro-F1 scores ranged between 93% and 97%, demonstrating the effectiveness of the calibration methodology.

# Usage
To use the interactive size predictor, simply run the last code cell of the notebook. You will be prompted to enter:

Weight (kg)
Age (years)
Height (cm)
The system will then output size recommendations for all defined brands with confidence scores. Type 'quit' at any prompt to exit the predictor.

# Dependencies
This project requires the following Python libraries:

os
warnings
pickle
numpy
pandas
matplotlib
seaborn
scipy
sklearn (Specifically MLPClassifier, StandardScaler, LabelEncoder, train_test_split, StratifiedKFold, Pipeline, classification_report, confusion_matrix, accuracy_score, f1_score, LogisticRegression, RandomForestClassifier, SVC, KNeighborsClassifier)
