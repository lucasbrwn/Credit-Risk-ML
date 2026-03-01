
import pandas as pd

def clean(path="data/cs-training.csv"):
    """
    Load and clean the dataset:
        1) Removes index column
        2) Fill missing values with median
        3) Remove extreme outliers to reduce skew
    """
    df = pd.read_csv(path)

    # Remove auto generated index values from columnns
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Fill missing values with median values
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].median())

    # Check Monthly Income for upper bound outliers
    upper_limit = df['MonthlyIncome'].quantile(0.99)
    df['MonthlyIncome'] = df['MonthlyIncome'].clip(upper=upper_limit)

    # Check DebtRatio for upper bounds outliers
    upper_dr = df['DebtRatio'].quantile(0.99)
    df['DebtRatio'] = df['DebtRatio'].clip(upper=upper_dr)
    df['DebtRatio'] = df['DebtRatio'].clip(upper=5)

    return df