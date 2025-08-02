import pandas as pd
from sklearn.linear_model import LinearRegression

file_path = "Samples/OneHotEncoding/homeprices.csv"

df = pd.read_csv(file_path)


# using Pandas to create dummy variables

dummies = pd.get_dummies(df.town)

merged_df = pd.concat([df, dummies], axis='columns')

#Dummy Variable Trap
#When you can derive one variable from other variables,
# they are known to be multi-colinear.
# In this situation linear regression won't work as expected.
# Hence we need to drop one column.

final_df = merged_df.drop(['town','west windsor'], axis='columns')
print(final_df.head())

model = LinearRegression()
X = final_df.drop('price', axis='columns')
y = final_df['price']
model.fit(X, y)
y_pred = model.predict(X)
print(y_pred)
print(final_df['price'])
print(model.score(X, y))


# let's use one Hot Encoding now

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
df_le = df
# df_le.town = le.fit_transform(df_le.town)
# this has converted text values in town in integers

X = df_le[['town','area']].values
y = df_le[['price']].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Suppose you want to apply OneHotEncoding to column 0 (first column)
column_transformer = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), [0])
    ],
    remainder='passthrough'  # keeps the rest of the columns unchanged
)
X = column_transformer.fit_transform(X)

X = X[:, 1:]
model.fit(X, y)
y_pred = model.predict(X)

print(model.score(X, y))


print(df_le.head())


