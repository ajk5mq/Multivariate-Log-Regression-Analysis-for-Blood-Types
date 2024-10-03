import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load your data (make sure to replace the path with your actual file path)
# Assuming your data is in a CSV format, adjust accordingly if it's in Excel
df = pd.read_csv('/Users/icedkiminho/Desktop/3-6-24 Datasheet.csv')

# Clean column names (strip spaces)
df.columns = df.columns.str.strip()
print("Initial DataFrame:")
print(df)

# Create a binary dependent variable for case type
df['Case'] = (df['Case Type'] == 1).astype(int)  # 1 for case, 0 for control
print("\nDataFrame after creating binary 'Case' column:")
print(df)

# Fill NaN values with 0 (or choose another method based on your research design)
df.fillna(0, inplace=True)
print("\nDataFrame after filling NaN values with 0:")
print(df)

# Create dummy variables for blood types
df_dummies = pd.get_dummies(df[['Type A', 'Type B', 'Type AB', 'Type O']], drop_first=True)
print("\nDummy variables for blood types:")
print(df_dummies)

# Add the dummies to the original DataFrame
df = pd.concat([df.reset_index(drop=True), df_dummies.reset_index(drop=True)], axis=1)
print("\nDataFrame after adding dummy variables:")
print(df)

# Create binary variable for Rh type
df['Rh_Positive'] = (df['Rh Type'] == 1).astype(int)  # Adjust based on your data
print("\nDataFrame after creating binary 'Rh_Positive' column:")
print(df)

# Remove duplicate columns if any
df = df.loc[:, ~df.columns.duplicated()]
print("\nDataFrame after removing duplicate columns:")
print(df)

# Select the relevant columns for the logistic regression model
X_columns = df_dummies.columns.tolist() + ['Rh_Positive']
X = df[X_columns]
y = df['Case']

# Check for multicollinearity
correlation_matrix = df[X_columns].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Check unique values for each predictor in X
for column in X.columns:
    print(f"{column} unique values: {df[column].unique()}")

# Drop columns with constant values
X = X.loc[:, (X != 0).any(axis=0)]
print("\nX DataFrame after dropping columns with constant values:")
print(X)

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the logistic regression model only if there's data
if X.shape[0] > 0 and y.shape[0] > 0:
    try:
        model = sm.Logit(y, X)
        result = model.fit(maxiter=100, method='bfgs')  # Increase iterations and specify method
        # Print the summary of the regression results
        print("\nLogistic Regression Results:")
        print(result.summary())

        # Calculate odds ratios with 0.5 adjustment
        odds_ratios = np.exp(result.params + 0.5).round(2)  # Add 0.5 adjustment and round to two decimal places
        p_values = result.pvalues.round(3)  # Round p-values to three decimal places

        # Calculate 95% CI with 0.5 adjustment
        conf = 1.96  # 95% CI z-value
        lower_ci = (np.exp(result.params - conf * result.bse + 0.5)).round(2)  # Add 0.5 adjustment and round
        upper_ci = (np.exp(result.params + conf * result.bse + 0.5)).round(2)  # Add 0.5 adjustment and round

        # Create DataFrame for odds ratios and CIs
        odds_ratio_df = pd.DataFrame({
            'Odds Ratio': odds_ratios,
            'P-Value': p_values,
            'Lower CI (95%)': lower_ci,
            'Upper CI (95%)': upper_ci
        })
        
        print("\nOdds Ratios with 95% Confidence Intervals (with 0.5 adjustment):")
        print(odds_ratio_df)
    except Exception as e:
        print("Error fitting the model:", e)
else:
    print("Not enough data to fit the model.")
