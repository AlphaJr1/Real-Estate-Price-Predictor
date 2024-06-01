# Real-Estate-Price-Predictor
This repo showcases my third-semester coursework project where I used linear regression to predict real estate prices. It includes data analysis, processing, and model building. Check out the notebook to see the process and results. Perfect for those interested in data science and real estate pricing. Enjoy! 🏡📈

## Project Overview

This project aims to predict the prices of real estate properties based on various features such as location, size, number of bedrooms, etc. The main steps involved are data analysis, data processing, and model development using linear regression.

## Getting Started

### Prerequisites

Before you get started, make sure you have the following installed:

- Python 3.x
- Jupyter Notebook
- Necessary Python libraries (you can install them using the `requirements.txt` file)

### Installation

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/AlphaJr1/Real-Estate-Price-Predictor.git
    ```

2. Navigate to the project directory:

    ```bash
    cd real-estate-price-prediction
    ```

3. Install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

4. Open the Jupyter Notebook:

    ```bash
    jupyter notebook
    ```

5. Open the `Test_RealestateLR.ipynb` file in Jupyter Notebook to explore the analysis and model development process.

## Project Structure

- `Test_RealestateLR.ipynb`: The Jupyter Notebook containing the data analysis and model development.
- `requirements.txt`: A list of required Python libraries.
- `data/`: (Optional) A folder to store your datasets if you plan to use different ones or update the existing data.

## Analysis

In the analysis section, I focused on understanding the dataset and extracting meaningful insights. Here are the key steps:

1. **Loading the Dataset**: I imported the dataset into the notebook and displayed the first few rows to get an initial look at the data.

    ```python
    import pandas as pd
    data = pd.read_csv('path_to_your_dataset.csv')
    data.head()
    ```

2. **Exploratory Data Analysis (EDA)**: I performed various analyses to understand the distribution of different features, the relationship between features, and how they influence the target variable (price). This included:
   - Plotting histograms and box plots to check the distribution and detect any outliers.
   - Using scatter plots and correlation matrices to identify relationships between features.
   - Summarizing statistics to get a sense of the central tendency and variability of the data.

    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Histogram
    data.hist(bins=50, figsize=(20,15))
    plt.show()

    # Correlation matrix
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True)
    plt.show()
    ```

## Data Processing

Data processing is a crucial step to prepare the data for modeling. Here’s what I did:

1. **Handling Missing Values**: I checked for any missing values in the dataset and decided on an appropriate strategy to handle them, such as filling them with the mean/median or dropping the rows/columns with missing values.

    ```python
    data = data.dropna()  # or data.fillna(data.mean(), inplace=True)
    ```

2. **Feature Engineering**: I created new features or modified existing ones to improve the model’s performance. For example, creating interaction terms or transforming skewed features.

    ```python
    data['new_feature'] = data['feature1'] * data['feature2']
    ```

3. **Encoding Categorical Variables**: I converted categorical variables into numerical values using techniques like one-hot encoding or label encoding.

    ```python
    data = pd.get_dummies(data, columns=['categorical_feature'])
    ```

4. **Splitting the Data**: I split the dataset into training and testing sets to evaluate the model's performance.

    ```python
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
    ```

## Linear Regression Model

For the model development, I implemented a linear regression model as follows:

1. **Model Building**: I used the `LinearRegression` class from `scikit-learn` to build the model. This involved fitting the model to the training data.

    ```python
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(train_set.drop('target', axis=1), train_set['target'])
    ```

2. **Model Training**: I trained the model using the training data and evaluated its performance on the testing data.

    ```python
    predictions = lin_reg.predict(test_set.drop('target', axis=1))
    ```

3. **Model Evaluation**: I assessed the model's performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²). These metrics helped in understanding how well the model is predicting the prices.

    ```python
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(test_set['target'], predictions)
    mse = mean_squared_error(test_set['target'], predictions)
    r2 = r2_score(test_set['target'], predictions)
    ```

4. **Improving the Model**: I iterated on the model by trying different features, tuning hyperparameters, and possibly using regularization techniques like Ridge or Lasso regression to improve the model's accuracy.

    ```python
    from sklearn.linear_model import Ridge, Lasso

    ridge_reg = Ridge(alpha=1, solver="cholesky")
    ridge_reg.fit(train_set.drop('target', axis=1), train_set['target'])

    lasso_reg = Lasso(alpha=0.1)
    lasso_reg.fit(train_set.drop('target', axis=1), train_set['target'])
    ```

## Conclusion

Through this project, I developed a linear regression model that can predict real estate prices with reasonable accuracy. The steps of data analysis, processing, and model development are clearly outlined in the notebook. Feel free to explore the notebook and see the detailed code and results.

## Contributing

Feel free to fork this repository, make changes, and submit pull requests. Any contributions to improve the model or add new features are welcome!

## Acknowledgements

A big thank you to all the open-source libraries and resources that made this project possible.
