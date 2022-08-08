# &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Customer Churn Prediction

#### Introduction:

Customer retention appears to be a crucial & fundamental necessity for every business, given the significance of customers as the most valuable assets of enterprises. Banks are no exception to this rule.  Customer retention is more important than ever given the competitive environment in which multiple banks are competing to provide various facilities including electronic banking services to stand out among the rest.

Carrying this notion in mind we are aiming to take the first step in this quest of finding the slips of the enterprise by finding out the customer churn rate (Churning refers to a customer who leaves one company to go to another company) which in turn will sketch the picture of how the enterprise is performing in current market scenario and then finding the reason of fallouts and hopefully implementing improvements to reverse the current situation. 

#### Dataset Description:

The dataset we are working with on for this project contains 12 features and about 10000 rows of customer data from a bank in Europe (specifically Spain, Germany & France).

   <img width="654" alt="image" src="https://user-images.githubusercontent.com/102252835/183488857-40f0c92b-beb5-4a71-bfb7-10c556647c2e.png">

The features in this dataset are as follows:
-	customer_id, (Numerical)
-	credit_score, (Numerical)
-	country, (Categorical) 
-	gender, (Categorical)
-	age, (Numerical)
-	tenure, (Numerical)
-	balance, (Numerical)
-	products_number, (Numerical)
-	credit_card, (Numerical)
-	active_member, (Numerical)
-	estimated_salary, (Numerical)
-	churn, (Target Label; 0 - No; 1 - yes)

The significance of these features are self-explanatory.

#### Expectation:

This is a classification problem where we are trying to classify and visualize the target lable ("Churn") from the dataset.Initially we will be performing feature selection using tree method (Random Forest) to work further on the selected features for this classification problem.

To accomplish this task we are leveraging various Supervised Machine Learning methods mentioned below:

- Naive Bayesian   
- SVM              
- XGBoost
- LightGBM
- Perceptron
- NeuralNets  

Thereafter, we plan to analyze and compare their performance which in turn will help us answer which among all are/is an efficient predictor for the given dataset, the feature that most affects the clients/users decision to leave and simultaneaously find the current churn rate of the enterprise and suggest improvements.

#### EDA:

- We have around `7,963` rows corresponding to Churn as **No**
- We have `2037` rows corresponding to Churn as **Yes**.
- We have the ratio of skewnesss `4:1` for every 4 values of churn as "No" we have, 1 value of Churn as "Yes". Therefore, we could see some skewness in the dataset.
![Churn Label Correlation](img/correlation_bank.png)
- From the correlation heatmap we could see that there is good correlation between `active-member - age`, `age - Churn` we need to take a look at more of the features more.

#### Goal:

- It is Imperative that the businesses take the right decisions and to maintain good customer base with good retention. This would lead to good business growth and finally affects the bottom-line for the company in the end.
- This dataset which contains the churn data, looks at all the features which might affect the customer churn rate and by applying smart ML algorithms, we could actually help predict whether the customer is retained or could leave the bank. 
- Through this, we could help the bank identify `Focus groups` in-order to target them and help the banks retain these customer.
