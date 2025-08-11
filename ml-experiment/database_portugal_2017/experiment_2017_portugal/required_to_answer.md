Overall goal is to decide and implement a method that successfully increase the resolution of the ERA5 Fire Weather Index (from its native ~25km to <=1km) for a test case in e.g., Portugal.

also make sure to verify the fwi with june 16
training data: jan1 to may1
validation: may1 to june1
TEST: june1 to july1

makesure to compare the results with the ERA5 FWI on the June 16 (disaster day). with our model enhanced FWI.


1. General analysis of the problem you are solving and what steps you will take to optimise
your model [10 marks]
a. Discussion of the features
b. Discussion of the number of features
c. Discussion of the feasibility of the task
Page 7 of 10
d. Discussion of the approach you will take
2. Data exploration and feature engineering [10 marks]
a. Distributions of data
b. Outliers
c. Correlations
d. Scaling
e. Transformations
f. Data balance
g. New features that could be generated
3. Model training [10 marks]
a. Performance measures 
- accuracy, precision, recall, F1 score, ROC-AUC
- Back-Aggregation Validation
- Spatial Correlation Analysis
- Cross-Scale Validation

b. Training/testing split
c. Cross-validation
d. Bias and Variance
4. Model optimisation [10 marks]
a. Hyperparameter tuning
5. Conclusions [10 marks]
a. Compare and contrast your algorithms' performance, identifying the best model and
why?
b. What additional steps could you take to improve your models?