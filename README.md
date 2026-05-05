# Traffic-accident-prediction


# Traffic Accident Prediction using Machine Learning


This project builds a machine learning model to predict traffic accident severity using structured road and environmental data. The goal is to identify patterns in accident occurrences and provide insights that can support road safety planning and risk assessment.

 Dataset

The dataset is sourced from Kaggle and contains 20,000 records with 20 features.

 Key Features

* **Location:** city, state, latitude, longitude
* **Time:** hour, day_of_week, is_weekend, is_peak_hour
* **Road Conditions:** road_type, lanes, traffic_signal
* **Environment:** weather, visibility, temperature
* **Traffic Factors:** traffic_density
* **Accident Details:** cause, vehicles_involved, casualties
* **Target Variable:** accident_severity


1. Data Preprocessing

* Data Cleaning
* Missing Value Imputation
* Categorical Encoding
* Feature Scaling
* Feature Engineering
* Data Balancing
* Train-Test Splitting

2. Model Building

A classification model is trained to predict accident severity levels based on input features.

 3. Model Evaluation

* Accuracy: **65%**
* Evaluation Metrics: Precision, Recall, F1-score, Confusion Matrix

> Note: Model performance is influenced by class imbalance and limited evaluation samples. Further improvements are required for better generalization.



 Key Observations

* The model performs better on the majority class
* Lower recall for minority classes indicates imbalance issues
* Further optimization is needed

 Feature Importance (Top 10)

* hour
* longitude
* latitude
* casualties
* temperature
* cause
* day_of_week
* vehicles_involved
* state
* lanes

 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

```bash
python main.py
```
 Future Improvements

* Handle class imbalance using advanced techniques (SMOTE, class weighting)
* Perform hyperparameter tuning
* Use cross-validation for robust evaluation
* Remove potential data leakage features
* Experiment with advanced models (XGBoost, Random Forest tuning)

 Conclusion

This project demonstrates a basic machine learning pipeline for traffic accident severity prediction. While initial results are moderate, the model provides a foundation for further improvements and real-world applications in traffic safety analytics.
