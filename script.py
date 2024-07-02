import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns

# load and investigate the data here:
df = pd.read_csv('tennis_stats.csv')
print(df.head())
# print(df.columns)

# perform exploratory analysis here:

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# print(numeric_cols)

# Function to perform single feature linear regression
def perform_linear_regression(feature_name):
    # Select feature and outcome
    X = df[[feature_name]]
    y = df['Winnings']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    # Initialize linear regression model
    model = LinearRegression()

    # Fit model on training data
    model.fit(X_train, y_train)

    # Evaluate model on test data
    score = model.score(X_test, y_test)
    print(f'R-squared score for {feature_name}: {score:.2f}')

    # Predict on test data
    y_pred = model.predict(X_test)

    # Plot actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.4)
    plt.xlabel('Actual Winnings')
    plt.ylabel('Predicted Winnings')
    plt.title(f'Actual vs Predicted Winnings for {feature_name}')
    plt.show()

# Perform linear regression for each numeric feature
for col in numeric_cols:
    if col != 'Winnings':
        perform_linear_regression(col)

## perform two feature linear regressions here:
print('Now performing Multi Linear Regression')

# Function to perform multiple linear regression
def perform_multi_linear_regression(X, y):
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    # Initialize linear regression model
    model = LinearRegression()

    # Fit model on training data
    model.fit(X_train, y_train)

    # Evaluate model on test data
    score = model.score(X_test, y_test)
    print(f'R-squared score: {score:.2f}')

    # Predict on test data
    y_pred = model.predict(X_test)

    # Plot actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.4)
    plt.xlabel('Actual Winnings')
    plt.ylabel('Predicted Winnings')
    plt.title('Actual vs Predicted Winnings')
    plt.show()

# Define the features and outcome
features = df[['BreakPointsOpportunities', 'FirstServeReturnPointsWon']]
outcome = df['Winnings']

# Perform multiple linear regression using the defined features and outcome
perform_multi_linear_regression(features, outcome)

## perform multiple feature linear regressions here:
features1 = df[['DoubleFaults', 'BreakPointsOpportunities']]
outcome1 = df['Winnings']
# Perform multiple linear regression using the defined features and outcome
perform_multi_linear_regression(features1, outcome1)

## perform multiple feature linear regressions here:
features2 = df[['FirstServePointsWon', 'SecondServePointsWon']]
outcome2 = df['Winnings']
# Perform multiple linear regression using the defined features and outcome
perform_multi_linear_regression(features2, outcome2)

## perform multiple feature linear regressions here:
features2 = df[['ReturnPointsWon', 'ServiceGamesWon']]
outcome2 = df['Winnings']
# Perform multiple linear regression using the defined features and outcome
perform_multi_linear_regression(features2, outcome2)

## perform multiple feature linear regressions here:
features3 = df[['Aces', 'BreakPointsOpportunities']]
outcome3 = df['Winnings']
# Perform multiple linear regression using the defined features and outcome
perform_multi_linear_regression(features3, outcome3)

## perform multiple feature linear regressions here:
features4 = df[['Wins', 'BreakPointsOpportunities']]
outcome4 = df['Winnings']
# Perform multiple linear regression using the defined features and outcome
perform_multi_linear_regression(features4, outcome4)


## perform multiple feature linear regressions here:
features5 = df[['ServiceGamesWon', 'BreakPointsOpportunities']]
outcome5 = df['Winnings']
# Perform multiple linear regression using the defined features and outcome
perform_multi_linear_regression(features5, outcome5)

