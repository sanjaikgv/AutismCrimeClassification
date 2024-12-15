## Overview
This project investigates crime patterns in Austin, Texas, using machine learning models to predict crime types based on temporal, spatial, and demographic factors. The study explores the effectiveness of Support Vector Machines (SVM), for classifying crime categories as part of the final project of the course INF 397 : Intro to Machine Learning with Statistical Analysis, Fall 2024.

The findings aim to support public safety initiatives by providing actionable insights into crime hotspots and patterns.

## Datasets
Data was sourced from the City of Austin Open Data Portal: https://data.austintexas.gov/

#### 1. Crime Reports Dataset: 
 - Link to dataset: https://data.austintexas.gov/Public-Safety/Crime-Reports/fdj4-gpfu/about_data
 - Time and location of crimes.
#### 2. NIBRS Crimes Dataset
 - Link to dataset: https://datahub.austintexas.gov/Public-Safety/NIBRS-Group-A-Offense-Crimes/i7fg-wrk5/about_data
 - Demographics and detailed crime descriptions.

## Data Preprocessing
- Merged datasets using Case Report Number.
- Filtered to crimes from Jan 1, 2019 - Sep 30, 2024.
- Dropped rare classes with fewer than 100 observations.
- Selected key features from importance metrics obtained from tree based and ensemble methods. 

## Final Data dimension - 
 - 100K observations and 35 predictors

## SVM Implementation
Utilized Support Vector Machines with both One-vs-One (OvO) and One-vs-Rest (OvR) classification strategies.

#### Approaches:

- Kernel Types: inear and radial
- Hyperparameter Tuning: Cost values (0.1, 1, 10).


#### Key Results:
- Linear Kernel (Tuned): Accuracy: **53.23%** | AUC: **51.46%**.
- Radial Kernel (OvO): Accuracy: **54.69%** | AUC: **51.97%**.
- Radial Kernel (OvR): Accuracy: **55.12%** | AUC: **52.37%**.

Insight: 
- Radial kernels slightly outperformed linear kernels, though at a higher computational cost.

## Challenges
- **Class Imbalance:** Dominance of specific crime types, such as assault.
  - Oversampling helps with this at the expense of accuracy. 
- **Computational Complexity:** High resource requirements for SVM training and cross-validation.
  - In RStudio, the svm() from the ‘e1071’ package trains the model in a single thread process, making it compute intensive to perform hyper-parameter tuning and cross-validation.
  - The time complexity of training an SVM is roughly O(n^3). 
  - `doParallel` and `foreach` were used to perform the
computation in multiple threads.
