# MachineLearning-TaxiServicePricePrediction

### FCIS's Machine Learning Course Project - Taxi Service Price Prediction, And Taxi Service Price Classification

---

The use of taxi service providers such as Uber, Kareem and Lyft has become almost essential in recent years. Each company has their own methods of pricing each ride. These prices may be affected by the locations or the weather. Given this dataset, our task is to predict the price of a taxi ride based on the provided information.

---

## Project Description:
- The objective of the project is to be prepared to apply different machine learning algorithms to real-world tasks. This will help me to increase my knowledge about the workflow of the machine learning tasks. I learned how to clean data, apply pre-processing, feature engineering, regression, and classification methods
- There are 2 large datasets. one is for regression, and the other is for classification
- There are small test datasets used for verification

---

#### Regression Part in README.md: https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/README.md#regression
#### Classification Part in README.md: https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/README.md#classification

---

## Regression

<img src="https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/Resources/reg1.jpg" alt="Info About The data" title="Info About The data" width="900" height="600">

### Steps Performed:
#### - Preprocessing:
- Used isnull() method to detect missing values and remove them using dropna() method
- Detected duplicated data using duplicated() method, and remove them using drop_duplicates()
- Detected outliers using boxplot() method in seaborn library, then remove them by using inter quartile range IQR method

<img src="https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/Resources/reg2.jpg" alt="Detecting Outliers" title="Detecting Outliers" width="900" height="600">

- Example:

<img src="https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/Resources/reg3.jpg" alt="With Outliers" title="With Outliers" width="400" height="600">&nbsp;<img src="https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/Resources/reg4.jpg" alt="Without Outliers" title="Without Outliers" width="400" height="600">

- Applied one hot-encoding technique on nominal features to make a good use of them. This was done by using get_dummies () method
- We have faced a Non Normal Distribution data represented in “temp“ feature in weather data set. So, we have used Median Absolute Deviation MAD method to overcome this problem
- We have overcome the problem of time_stamp feature by transforming these unix time into year-month-day mode by using: datetime.utcfromtimestamp(int(ts)).strftime('%Y-%m-%d')) method
- We have done another method to deal with it by keeping it in unix time stamp mode and only truncated specific numbers using astype(str).str[:7]

#### - Analysis:
- We have done some Exploratory data analysis and visualization, that show some information about our data set such as regplot(), groupby().plot().pie() and hist plot

<img src="https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/Resources/reg6.jpg" alt="Hist Plot" title="Hist Plot" width="400" height="600">&nbsp;<img src="https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/Resources/reg7.jpg" alt="groupby(['cab_type'])['source']" title="groupby(['cab_type'])['source']" width="400" height="600">&nbsp;<img src="https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/Resources/reg8.jpg" alt="groupby('destination')['distance']" title="groupby('destination')['distance']" width="400" height="600">

- We also applied pairplot() method to show how features affect each other and the target as well
- Also from visualization we used heatmap() which shows correlation between the features. As if the correlation between two feature is high so, it men that there is a redundant feature so we will drop one of them which have less correlation with the target

<img src="https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/Resources/reg10.jpg" alt="Correlation" title="Correlation" width="600" height="600">&nbsp;<img src="https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/Resources/reg9.jpg" alt="Correlation" title="Correlation" width="600" height="600">

- Then we merged the two datasets

<img src="https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/Resources/reg5.jpg" alt="Merging" title="Merging" width="900" height="600">

#### - Used Regression Techniques
1. Polynomial Regression<br>
&nbsp;&nbsp;- train error / Root_mean_squared_error :1.7682491756862755<br>
&nbsp;&nbsp;- train error / r2_score :0.9578699387248786<br>
&nbsp;&nbsp;- test error / Root_mean_squared_error :1.7594747267177582<br>
&nbsp;&nbsp;- test error / r2_score :0.9583861915850564

2. L2 Regularization Ridge<br>
&nbsp;&nbsp;- train error / Root_mean_squared_error :1.8686930938976944<br>
&nbsp;&nbsp;- train error / r2_score :0.9547317306195743<br>
&nbsp;&nbsp;- test error / Root_mean_squared_error :1.8440148895744963<br>
&nbsp;&nbsp;- test error / r2_score :0.9558746844549015

3. Random Forest Regressor<br>
&nbsp;&nbsp;- train error / Root_mean_squared_error :1.6031737243094235<br>
&nbsp;&nbsp;- train error / r2_score :0.965219197259477<br>
&nbsp;&nbsp;- test error / Root_mean_squared_error :1.584148011646326<br>
&nbsp;&nbsp;- test error / r2_score :0.967017157507762

4. XGBoost<br>
&nbsp;&nbsp;- train error / Root_mean_squared_error :1.7345250930795648<br>
&nbsp;&nbsp;- train error / r2_score :0.9609330219794602<br>
&nbsp;&nbsp;- test error / Root_mean_squared_error :1.7402267667358167<br>
&nbsp;&nbsp;- test error / r2_score :0.9605351782969276

---
---

## Classification

<img src="https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/Resources/class1.jpg" alt="Info About The data" title="Info About The data" width="900" height="600">

### Steps Performed:
#### - Preprocessing:
- Used isnull() method to detect missing values and remove them using dropna() method
- Detected duplicated data using duplicated() method, and remove them using drop_duplicates()
- Detected outliers using boxplot() method in seaborn library, then remove them by using inter quartile range IQR method
- Example:

<img src="https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/Resources/class2.jpg" alt="With Outliers" title="With Outliers" width="400" height="600">&nbsp;<img src="https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/Resources/class3.jpg" alt="Without Outliers" title="Without Outliers" width="400" height="600">

- Applied one hot-encoding technique on nominal features to make a good use of them. This was done by using get_dummies () method
- We have faced a Non Normal Distribution data represented in “temp“ feature in weather data set. So, we have used Median Absolute Deviation MAD method to overcome this problem
- We have overcome the problem of time_stamp feature by transforming these unix time into year-month-day mode by using: datetime.utcfromtimestamp(int(ts)).strftime('%Y-%m-%d')) method
- We have done another method to deal with it by keeping it in unix time stamp mode and only truncated specific numbers using astype(str).str[:7]

#### - Analysis:
- We have done some Exploratory data analysis and visualization, that show some information about our data

<img src="https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/Resources/class4.jpg" alt="Count Plot" title="Count Plot" width="400" height="600">&nbsp;<img src="https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/Resources/class5.jpg" alt="Plot" title="Plot" width="400" height="600">&nbsp;<img src="https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/Resources/class6.jpg" alt="Plot" title="Plot" width="400" height="600">

- Then we merged the two datasets

#### - Used Classification Techniques
1. Decision Tree<br>
&nbsp;&nbsp;- train accuracy: 90.26026194346353%<br>
&nbsp;&nbsp;- test accuracy: 90.1370426074528%

<img src="https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/Resources/class10.jpg" alt="Decision Tree" title="Decision Tree" width="900" height="600">

2. Random Forest<br>
&nbsp;&nbsp;- train accuracy: 89.79153821721503%<br>
&nbsp;&nbsp;- test accuracy: 89.83063514157332%

<img src="https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/Resources/class7.jpg" alt="Random Forest" title="Random Forest" width="900" height="600">

3. AdaBoost<br>
&nbsp;&nbsp;- train accuracy: 90.40926330736319%<br>
&nbsp;&nbsp;- test accuracy: 89.86706773660168%

<img src="https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/Resources/class8.jpg" alt="AdaBoost" title="AdaBoost" width="900" height="600">

4. Logistic Regression<br>
&nbsp;&nbsp;- train accuracy: 88.87230723240475%<br>
&nbsp;&nbsp;- test accuracy: 88.99735630143769%

<img src="https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/Resources/class9.jpg" alt="Logistic Regression" title="Logistic Regression" width="900" height="600">

#### - Hyperparameter Tuning
- <img src="https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/Resources/class11.jpg" alt="Hyperparameter Tuning" title="Hyperparameter Tuning" width="900" height="600">
- <img src="https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/Resources/class13.jpg" alt="Hyperparameter Tuning" title="Hyperparameter Tuning" width="900" height="600">
- <img src="https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/Resources/class12.jpg" alt="Hyperparameter Tuning" title="Hyperparameter Tuning" width="900" height="600">
- <img src="https://github.com/Amr-Wael-Dev/MachineLearning-TaxiServicePricePrediction/blob/main/Resources/class14.jpg" alt="Hyperparameter Tuning" title="Hyperparameter Tuning" width="900" height="600">
