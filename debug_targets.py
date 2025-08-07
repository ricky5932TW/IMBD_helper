import pandas as pd
import numpy as np

# Load the data to check target variable distribution
all_train_data = pd.read_csv(rf'C:\Users\E4-159\Documents\py_surr\imbd2025\初\final_combined_data.csv')

# Check target variables
y_x = all_train_data['Disp. X']
y_z = all_train_data['Disp. Z']

print("=== Target Variable Analysis ===")
print(f"y_x (Disp. X) statistics:")
print(y_x.describe())
print(f"\ny_x range: {y_x.min()} to {y_x.max()}")
print(f"y_x std: {y_x.std()}")

print(f"\ny_z (Disp. Z) statistics:")
print(y_z.describe())
print(f"\ny_z range: {y_z.min()} to {y_z.max()}")
print(f"y_z std: {y_z.std()}")

# Check for NaN values
print(f"\ny_x NaN count: {y_x.isnull().sum()}")
print(f"y_z NaN count: {y_z.isnull().sum()}")

# Check data types
print(f"\ny_x dtype: {y_x.dtype}")
print(f"y_z dtype: {y_z.dtype}")

# Check first few values
print(f"\nFirst 10 y_x values:")
print(y_x.head(10).values)

print(f"\nFirst 10 y_z values:")
print(y_z.head(10).values)
