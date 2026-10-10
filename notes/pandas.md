# 04 — Pandas Notes for Machine Learning

## 1. What Is Pandas?

**Pandas** is a Python library used to load, inspect, clean, transform, and analyze structured data.

It is especially useful for working with CSV files, Excel files, and tabular datasets.

### Why Pandas is important for ML

- Load datasets into Python.
- Inspect rows, columns, and data types.
- Handle missing values.
- Select and filter data.
- Detect duplicate records.
- Calculate statistics and group data.
- Prepare features and targets before model training.

### Installation and Import

Install Pandas:

```bash
python -m pip install pandas
```

Import the library:

```python
import pandas as pd
```

`pd` is the conventional alias for Pandas.

---

## 2. Series and DataFrame

Pandas has two main data structures.

### Series

A Series is a one-dimensional labeled collection of values.

```python
import pandas as pd

marks = pd.Series([80, 75, 90, 85])

print(marks)
```

Output:

```text
0    80
1    75
2    90
3    85
dtype: int64
```

The left side contains index labels; the right side contains values.

### DataFrame

A DataFrame is a two-dimensional table with rows and columns.

```python
df = pd.DataFrame({
    "Name": ["Aman", "Riya", "Rahul"],
    "Age": [20, 21, 19],
    "Marks": [80, 90, 75]
})

print(df)
```

Output:

```text
    Name  Age  Marks
0   Aman   20     80
1   Riya   21     90
2  Rahul   19     75
```

A DataFrame is similar to a spreadsheet or database table.

---

## 3. Loading and Saving Data

### Read a CSV File

```python
df = pd.read_csv("students.csv")
```

Pandas reads the file into a DataFrame.

### Read an Excel File

```python
df = pd.read_excel("students.xlsx")
```

Reading Excel files may require an additional engine such as `openpyxl`.

### Save to CSV

```python
df.to_csv("output.csv", index=False)
```

`index=False` prevents Pandas from writing the DataFrame's index as an extra column.

### Save to Excel

```python
df.to_excel("output.xlsx", index=False)
```

### Useful Parameters

```python
df = pd.read_csv(
    "students.csv",
    usecols=["Name", "Marks"],
    nrows=100
)
```

- `usecols`: loads selected columns.
- `nrows`: loads a limited number of rows.
- `index=False`: avoids writing the index when exporting.

---

## 4. Inspecting a Dataset

After loading data, inspect it before modifying anything.

Assume `df` is your DataFrame.

### First Rows

```python
print(df.head())
```

Displays the first five rows by default.

```python
print(df.head(10))
```

Displays the first ten rows.

### Last Rows

```python
print(df.tail())
```

Displays the last five rows by default.

### Dataset Shape

```python
print(df.shape)
```

Example output:

```text
(1000, 5)
```

This means 1,000 rows and 5 columns.

### Column Names

```python
print(df.columns)
```

### Data Types and Non-Missing Counts

```python
df.info()
```

This shows column names, data types, non-null counts, and memory usage.

### Statistical Summary

```python
print(df.describe())
```

By default, this summarizes numerical columns.

```python
print(df.describe(include="all"))
```

Includes summaries for categorical columns as well.

### Data Types

```python
print(df.dtypes)
```

Common types include:
- `int64`: integers
- `float64`: floating-point numbers
- `object`: often strings or mixed values
- `bool`: Boolean values
- `datetime64`: date/time values

Pandas may use different or nullable data types depending on the data and version.

---

## 5. Selecting Columns and Rows

Consider this example dataset:

```python
df = pd.DataFrame({
    "Name": ["Aman", "Riya", "Rahul", "Neha"],
    "Age": [20, 21, 19, 22],
    "Marks": [80, 90, 75, 95]
})
```

### Select One Column

```python
print(df["Name"])
```

Returns a Series.

### Select Multiple Columns

```python
print(df[["Name", "Marks"]])
```

Returns a DataFrame.

### Select Rows Using `loc`

`loc` selects by index labels.

```python
print(df.loc[0])
print(df.loc[0:2])
```

For a standard integer index, `df.loc[0:2]` includes index label `2`.

### Select Rows Using `iloc`

`iloc` selects by integer position.

```python
print(df.iloc[0])
print(df.iloc[0:2])
```

`df.iloc[0:2]` selects positions 0 and 1; the stop position is excluded.

### Select a Specific Cell

```python
print(df.loc[1, "Name"])
print(df.iloc[1, 0])
```

Both return `"Riya"` for this dataset.

**Remember:**
- `loc` = index labels.
- `iloc` = integer positions.

---

## 6. Filtering Rows

Filtering selects rows that satisfy a condition.

### One Condition

```python
print(df[df["Marks"] > 80])
```

Returns rows where marks exceed 80.

### Multiple Conditions

Use `&` for AND and `|` for OR. Put each condition inside parentheses.

```python
print(df[(df["Age"] > 19) & (df["Marks"] >= 80)])
```

### OR Condition

```python
print(df[(df["Age"] < 20) | (df["Marks"] >= 90)])
```

### Using `isin()`

Select rows where a column matches one of several values.

```python
print(df[df["Name"].isin(["Aman", "Neha"])])
```

### Using `between()`

```python
print(df[df["Marks"].between(80, 90)])
```

The endpoints are included by default.

---

## 7. Adding, Updating, and Removing Columns

### Create a Column

```python
df["Passed"] = df["Marks"] >= 40
```

Creates a Boolean column.

### Calculate a Column from Existing Columns

```python
df["Bonus_Marks"] = df["Marks"] + 5
```

### Update Values

```python
df.loc[df["Marks"] < 80, "Marks"] = 80
```

This sets marks below 80 to 80.

### Rename Columns

```python
df = df.rename(columns={
    "Name": "Student_Name",
    "Marks": "Score"
})
```

### Drop a Column

```python
df = df.drop(columns=["Bonus_Marks"])
```

### Drop Rows

```python
df = df.drop(index=[0, 1])
```

These operations return a modified DataFrame by default. Assign the result back to `df` if you want to keep the changes.

---

## 8. Sorting Data

### Sort by One Column

```python
df = df.sort_values("Marks")
```

Sorts in ascending order by default.

### Descending Order

```python
df = df.sort_values("Marks", ascending=False)
```

### Sort by Multiple Columns

```python
df = df.sort_values(
    ["Age", "Marks"],
    ascending=[True, False]
)
```

This sorts by age ascending, then marks descending within equal ages.

---

## 9. Missing Values

Missing values are common in real-world datasets.

Pandas often represents missing values as `NaN` or `None`. Nullable data types can also use `pd.NA`.

### Detect Missing Values

```python
print(df.isna())
```

Returns a Boolean DataFrame.

### Count Missing Values per Column

```python
print(df.isna().sum())
```

### Count All Missing Values

```python
print(df.isna().sum().sum())
```

### Remove Rows with Missing Values

```python
df = df.dropna()
```

Use this only when dropping those rows is appropriate.

### Fill Missing Values

```python
df["Age"] = df["Age"].fillna(df["Age"].median())
```

For a categorical column:

```python
df["City"] = df["City"].fillna("Unknown")
```

### Mean vs Median Imputation

- **Mean:** Can work for numerical data without strong outlier effects.
- **Median:** Often preferable when numerical data contains extreme values.
- **Mode or a placeholder:** May be appropriate for categorical data, depending on the meaning of missingness.

**ML rule:** When preprocessing data for model evaluation, calculate imputation values from the training set only. A scikit-learn `Pipeline` helps enforce this correctly.

---

## 10. Duplicate Values

Duplicates may distort statistics and model training.

### Detect Duplicates

```python
print(df.duplicated())
```

Returns `True` for duplicate rows after the first occurrence.

### Count Duplicates

```python
print(df.duplicated().sum())
```

### Remove Duplicates

```python
df = df.drop_duplicates()
```

### Check Duplicates Based on Specific Columns

```python
print(df.duplicated(subset=["Name"]))
```

Two rows can be duplicates according to a chosen column even if other columns differ. Confirm whether removing them makes sense for the problem.

---

## 11. Data Types and Conversion

Incorrect data types can cause errors in analysis and ML.

### Convert a Column to Integer

```python
df["Age"] = df["Age"].astype("int64")
```

This works only if the values can be converted without invalid or missing values.

### Convert Text to Numbers Safely

```python
df["Marks"] = pd.to_numeric(
    df["Marks"],
    errors="coerce"
)
```

Invalid values become missing values (`NaN`).

### Convert to Datetime

```python
df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)
```

Invalid dates become `NaT`, which represents a missing datetime.

**Important:** After conversion, inspect missing values because invalid data may have been converted into missing entries.

---

## 12. String Operations

Use `.str` to work with text columns.

```python
df["Name"] = df["Name"].str.strip()
```

Removes leading and trailing whitespace.

```python
df["Name"] = df["Name"].str.lower()
```

Converts text to lowercase.

```python
df["Name"] = df["Name"].str.upper()
```

Converts text to uppercase.

```python
df[df["Name"].str.contains("rah", case=False, na=False)]
```

Selects rows where the name contains `"rah"`, ignoring case and treating missing strings as non-matches.

String cleaning is useful when categories contain inconsistent capitalization or extra spaces.

---

## 13. Applying Functions

### `apply()`

`apply()` applies a function along a Series or DataFrame axis.

```python
df["Marks_Doubled"] = df["Marks"].apply(lambda x: x * 2)
```

### Prefer Vectorized Operations When Possible

Instead of:

```python
df["Marks_Doubled"] = df["Marks"].apply(lambda x: x * 2)
```

Prefer:

```python
df["Marks_Doubled"] = df["Marks"] * 2
```

Vectorized operations are usually clearer and often faster for straightforward calculations.

---

## 14. Grouping and Aggregation

`groupby()` divides rows into groups and calculates summaries for each group.

```python
df = pd.DataFrame({
    "Department": ["CSE", "ECE", "CSE", "ECE", "CSE"],
    "Marks": [80, 70, 90, 75, 85]
})

print(df.groupby("Department")["Marks"].mean())
```

Output:

```text
Department
CSE    85.0
ECE    72.5
Name: Marks, dtype: float64
```

### Multiple Aggregations

```python
print(
    df.groupby("Department")["Marks"].agg(
        ["mean", "min", "max", "count"]
    )
)
```

This calculates the mean, minimum, maximum, and count for each department.

### Group by Multiple Columns

```python
df.groupby(["Department", "Year"])["Marks"].mean()
```

This requires a `Year` column to exist in the DataFrame.

**Why it matters:** Grouping helps analyze performance, sales, customer behavior, and other patterns before modeling.

---

## 15. Basic Statistics

### Mean, Median, and Mode

```python
print(df["Marks"].mean())
print(df["Marks"].median())
print(df["Marks"].mode())
```

`mode()` can return more than one value if multiple values tie for the highest frequency.

### Minimum and Maximum

```python
print(df["Marks"].min())
print(df["Marks"].max())
```

### Count Values

```python
print(df["Department"].value_counts())
```

Counts how often each category occurs.

```python
print(df["Department"].value_counts(normalize=True))
```

Returns proportions instead of raw counts.

### Correlation

```python
print(df[["Age", "Marks"]].corr())
```

Correlation measures the strength and direction of a linear relationship between numerical columns.

A correlation near `1` indicates a strong positive linear relationship; near `-1` indicates a strong negative linear relationship; near `0` indicates little linear relationship.

**Important:** Correlation does not prove causation, and it does not necessarily capture nonlinear relationships.

---

## 16. Combining DataFrames

### Concatenation

`concat()` joins DataFrames along rows or columns.

```python
df1 = pd.DataFrame({"Name": ["Aman"], "Marks": [80]})
df2 = pd.DataFrame({"Name": ["Riya"], "Marks": [90]})

result = pd.concat([df1, df2], ignore_index=True)
print(result)
```

Output:

```text
   Name  Marks
0  Aman     80
1  Riya     90
```

`ignore_index=True` creates a new sequential index.

### Merge

`merge()` combines DataFrames using matching key columns, similar to a SQL join.

```python
students = pd.DataFrame({
    "Student_ID": [1, 2],
    "Name": ["Aman", "Riya"]
})

marks = pd.DataFrame({
    "Student_ID": [1, 2],
    "Marks": [80, 90]
})

result = pd.merge(students, marks, on="Student_ID")
print(result)
```

Output:

```text
   Student_ID  Name  Marks
0           1  Aman     80
1           2  Riya     90
```

Common merge types:
- `inner`: only matching keys.
- `left`: all left rows and matching right rows.
- `right`: all right rows and matching left rows.
- `outer`: all keys from both DataFrames.

Example:

```python
pd.merge(students, marks, on="Student_ID", how="left")
```

Check whether join keys are unique when you expect one row per entity. Duplicate keys can unexpectedly multiply rows during a merge.

---

## 17. Indexing

The DataFrame index labels rows. They are not necessarily the same as row positions.

### Set an Index

```python
df = df.set_index("Student_ID")
```

### Reset an Index

```python
df = df.reset_index()
```

### Important Difference

After filtering, the original index labels may remain unchanged.

```python
filtered = df[df["Marks"] >= 80]
```

If you need a fresh sequential index:

```python
filtered = filtered.reset_index(drop=True)
```

You do not need to reset the index after every operation. Do so only when useful.

---

## 18. Exporting and Inspecting a Clean Dataset

A simple workflow:

```python
import pandas as pd

df = pd.read_csv("students.csv")

print(df.head())
print(df.shape)
df.info()

print(df.isna().sum())
print(df.duplicated().sum())

df = df.drop_duplicates()

df.to_csv("cleaned_students.csv", index=False)
```

This example checks the dataset, removes exact duplicate rows, and exports the result. It does not automatically resolve missing values, incorrect data types, or domain-specific problems.

Cleaning requires understanding the dataset, not blindly applying every available method.

---

## 19. Pandas in Machine Learning

Pandas is commonly used to prepare a dataset before passing it to scikit-learn.

### Separate Features and Target

Suppose the target column is `Price`.

```python
X = df.drop(columns=["Price"])
y = df["Price"]
```

- `X`: feature DataFrame.
- `y`: target Series.

For classification, `y` might contain categories such as `"spam"` and `"not spam"`. For regression, it might contain numerical prices.

### Split Before Fitting Preprocessing

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This creates training and test subsets.

For classification, you may use `stratify=y` when appropriate. For time-series data, use a time-aware split rather than randomly mixing past and future observations.

### Pandas vs NumPy

| Pandas | NumPy |
|---|---|
| Works naturally with labeled rows and columns | Works naturally with numerical arrays |
| Has Series and DataFrame | Has `ndarray` |
| Useful for cleaning and inspecting tabular data | Useful for numerical and matrix operations |
| Supports column names and index labels | Uses positional array indexing |

Pandas often uses NumPy internally, but the two libraries serve different purposes.

---

## 20. Common Mistakes

1. Confusing `loc` with `iloc`.
2. Forgetting that filtering may preserve the original index.
3. Using `and`/`or` instead of `&`/`|` for multiple Pandas conditions.
4. Forgetting parentheses around individual conditions.
5. Ignoring missing values and duplicates.
6. Converting invalid text to numbers without checking the resulting missing values.
7. Using `apply()` when a simple vectorized operation is clearer.
8. Assuming a merge always produces one output row per input row.
9. Calculating preprocessing statistics using the full dataset before splitting.
10. Treating correlation as proof of causation.
11. Dropping data without understanding why it is missing or duplicated.
12. Forgetting `index=False` when exporting a CSV if the index should not become a column.

---

## 21. Quick Revision Cheat Sheet

| Task | Pandas syntax |
|---|---|
| Read CSV | `pd.read_csv("data.csv")` |
| Read Excel | `pd.read_excel("data.xlsx")` |
| First rows | `df.head()` |
| Dataset dimensions | `df.shape` |
| Column names | `df.columns` |
| Data types and non-null counts | `df.info()` |
| Summary statistics | `df.describe()` |
| Select column | `df["Age"]` |
| Select multiple columns | `df[["Age", "Marks"]]` |
| Select by label | `df.loc[0:2]` |
| Select by position | `df.iloc[0:2]` |
| Filter rows | `df[df["Marks"] > 80]` |
| Missing values | `df.isna().sum()` |
| Fill missing values | `df["Age"].fillna(df["Age"].median())` |
| Remove duplicates | `df.drop_duplicates()` |
| Sort | `df.sort_values("Marks")` |
| Count categories | `df["City"].value_counts()` |
| Group and aggregate | `df.groupby("City")["Marks"].mean()` |
| Combine DataFrames | `pd.concat([df1, df2])` |
| Merge on a key | `pd.merge(df1, df2, on="ID")` |
| Convert to numeric | `pd.to_numeric(df["Age"], errors="coerce")` |
| Separate features | `X = df.drop(columns=["Target"])` |
| Separate target | `y = df["Target"]` |
| Export CSV | `df.to_csv("output.csv", index=False)` |

---

## 22. Practice Exercises

Use a small student dataset or create your own DataFrame.

1. Create a DataFrame containing student names, ages, departments, and marks.
2. Display the first three rows and inspect its shape and data types.
3. Select only the name and marks columns.
4. Filter students who scored above 80.
5. Filter students who are at least 20 years old and scored above 70.
6. Find missing values in every column.
7. Fill missing numerical values with the median.
8. Find and remove duplicate rows.
9. Sort students by marks in descending order.
10. Calculate the average marks for each department.
11. Count students in each department.
12. Merge student details with a second DataFrame containing attendance.
13. Save the cleaned data to a CSV file.
14. Separate features and target, then split them into training and test sets.

**Completion goal:** You should be able to load a dataset, inspect it, select and filter columns, handle missing values, group and combine data, and prepare `X` and `y` for ML.

