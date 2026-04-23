import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
hello
# -----------------------------
# 1. LOAD DATASET
# -----------------------------
df = pd.read_csv("stock_market_data.csv", header=[0,1], index_col=0)

# -----------------------------
# 2. CLEAN DATA
# -----------------------------
df.columns = ['_'.join(col) for col in df.columns]
df.index = pd.to_datetime(df.index)
df.reset_index(inplace=True)
df.rename(columns={'index': 'Date'}, inplace=True)

print("rows:", len(df))
print("columns:", len(df.columns))

print("\nmissing values:\n", df.isnull().sum())

df = df.drop_duplicates()

# -----------------------------
# 3. EDA
# -----------------------------
print("\nSummary Statistics:\n", df.describe())
print("\nCorrelation Matrix:\n", df.corr(numeric_only=True))

# -----------------------------
# OUTLIER DETECTION (IQR)
# -----------------------------
num_cols = ['Close_AAPL','Close_MSFT','Close_TSLA']

print("\nIQR Outlier Detection")
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(col, "outliers:", len(outliers))

# -----------------------------
# OBJECTIVE 1: CORRELATION
# -----------------------------
corr = df['Close_AAPL'].corr(df['Close_TSLA'])
print("\nCorrelation (AAPL vs TSLA):", corr)

# -----------------------------
# OBJECTIVE 2: COMPANY COMPARISON
# -----------------------------
print("\nAverage Closing Prices:\n")
print("AAPL:", df['Close_AAPL'].mean())
print("MSFT:", df['Close_MSFT'].mean())
print("TSLA:", df['Close_TSLA'].mean())

# -----------------------------
# OBJECTIVE 3: VISUALIZATION
# -----------------------------

# 1. Scatter Plot
plt.figure()
sns.scatterplot(x=df['Close_AAPL'], y=df['Close_TSLA'])
plt.title("AAPL vs TSLA")
plt.show()

# 2. Histogram
plt.figure()
sns.histplot(df['Close_TSLA'], kde=True)
plt.title("TSLA Price Distribution")
plt.show()

#  3. Line Plot (Date vs Price)
plt.figure()

plt.plot(df['Date'], df['Close_TSLA'], marker='o')

plt.title("TSLA Price Trend")
plt.xlabel("Date")
plt.ylabel("Price")

plt.show()

# 4. Bubble Plot
plt.figure()
plt.scatter(df['Close_AAPL'], df['Close_TSLA'], 
            s=df['Close_MSFT'], alpha=0.5)
plt.title("Bubble Plot (AAPL vs TSLA)")
plt.xlabel("AAPL")
plt.ylabel("TSLA")
plt.show()

# 5. Box Plot
plt.figure()
sns.boxplot(data=df[['Close_AAPL','Close_MSFT','Close_TSLA']])
plt.title("Stock Price Distribution")
plt.show()

# 6. Bar Plot
means = [df['Close_AAPL'].mean(), df['Close_MSFT'].mean(), df['Close_TSLA'].mean()]
labels = ['AAPL','MSFT','TSLA']

plt.figure()
plt.bar(labels, means)
plt.title("Average Stock Prices")
plt.show()

# 7. Heatmap
plt.figure()
sns.heatmap(df[['Close_AAPL','Close_MSFT','Close_TSLA']].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# 8. Pie Chart
plt.figure()
values = [df['Close_AAPL'].mean(), df['Close_MSFT'].mean(), df['Close_TSLA'].mean()]
plt.pie(values, labels=labels, autopct='%1.1f%%')
plt.title("Stock Distribution")
plt.show()

# -----------------------------
# OBJECTIVE 4: HYPOTHESIS TESTING
# -----------------------------
t_stat, p_val = ttest_1samp(df['Close_TSLA'], df['Close_TSLA'].mean())

print("\nt-test p-value:", p_val)

mean = df['Close_TSLA'].mean()
std = df['Close_TSLA'].std()
n = len(df)

z = (mean - mean) / (std / np.sqrt(n))
print("z-score:", z)

# -----------------------------
# OBJECTIVE 5: LINEAR REGRESSION
# -----------------------------
x = df['Close_AAPL']
y = df['Close_TSLA']

mean_x = x.mean()
mean_y = y.mean()

num = ((x - mean_x) * (y - mean_y)).sum()
den = ((x - mean_x) ** 2).sum()

slope = num / den
intercept = mean_y - slope * mean_x

print("\nRegression Equation:")
print("y =", slope, "* x +", intercept)

plt.figure()
sns.regplot(x=x, y=y)
plt.title("Regression Plot (AAPL vs TSLA)")
plt.show()
