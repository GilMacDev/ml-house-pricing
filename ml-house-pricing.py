import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Forcasting problem
# Here we are predicting the sale prices of houses based on various features
# The data comes from the Kaggle House Prices: Advanced Regression Techniques competition

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Display nulls
print("Initial Null Values in Train Dataset:", train.isnull().sum().sum())
print("Initial Null Values in Test Dataset:", test.isnull().sum().sum())

# Separate out Saleprice as it is the target variable
y = train["SalePrice"]
X_train = train.drop("SalePrice", axis=1)

# Add a 'Type' column to differentiate between train and test data
X_train["Type"] = "train"
test["Type"] = "test"

# Combine datasets to streamline data preprocessing
data = pd.concat([X_train, test], ignore_index=True)

# Map columns with nulls to replacement values
null_replacements = {
    "Electrical": "SBrkr",
    "MSZoning": "RL",
    "LotFrontage": data["LotFrontage"].mean(),
    "Alley": "Nothing",
    "Utilities": "AllPub",
    "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd",
    "MasVnrArea": 0,
    "MasVnrType": "None",
    "BsmtCond": "No",
    "BsmtExposure": "NB",
    "BsmtFinType1": "NB",
    "BsmtFinSF1": 0.0,
    "BsmtFinSF2": 0.0,
    "BsmtUnfSF": 0.0,
    "TotalBsmtSF": 0.0,
    "BsmtFullBath": 0.0,
    "BsmtHalfBath": 0.0,
    "KitchenQual": "TA",
    "Functional": "Typ",
    "FireplaceQu": "None",
    "GarageType": "No",
    "GarageYrBlt": 0,
    "GarageFinish": "No",
    "GarageCars": 0,
    "GarageArea": 0,
    "GarageQual": "No",
    "GarageCond": "No",
    "PoolQC": "No",
    "Fence": "No",
    "MiscFeature": "No",
    "SaleType": "Con",
    "SaleCondition": "None",
    "BsmtQual": "TA",
    "BsmtFinType2": "Unf",
}

# Replacing null values
data.fillna(null_replacements, inplace=True)

# Check for remaining null values
print("Remaining Null Values After Replacement:", data.isnull().sum().sum())

# Creating TimeIndex feature that combines year and month sold
data["TimeIndex"] = data["YrSold"].astype(str) + data["MoSold"].astype(str).str.pad(
    2, fillchar="0"
)

# Convert TimeIndex to integer for clustering
data["TimeIndex"] = data["TimeIndex"].astype(int)

# Encode Labels
int_columns = data[data.columns[data.dtypes == "int"]]
object_columnns = data[data.columns[data.dtypes == "object"]]
float_columns = data[data.columns[data.dtypes == "float"]]

for i in object_columnns:
    if i != "Type" and i != "TimeIndex":
        label = LabelEncoder()
        label.fit(data[i].values)
        data[i] = label.transform(data[i].values)

# Defining forecasting features
forecasting_features = [
    "Neighborhood",
    "OverallQual",
    "YearBuilt",
    "GrLivArea",
    "LotArea",
    "TimeIndex",
]

# Apply unsupervised KMeans clustering on the forecasting features
kmeans = KMeans(n_clusters=4, n_init=10, random_state=1)
data["Cluster"] = kmeans.fit_predict(data[forecasting_features])

# Split data
train_cleaned = data[data["Type"] == "train"].drop("Type", axis=1)
test_cleaned = data[data["Type"] == "test"].drop("Type", axis=1)

# Arrays to hold results
score_test_r2 = []
score_test_mse = []
model = []

# Testing two train test splits to compare results

# 90% of data is used for training, 10% is for testing
x_train_90, x_test_10, y_train_90, y_test_10 = train_test_split(
    train_cleaned, y, test_size=0.10, random_state=1
)

# 75% of data is used for training, 25% is used for testing
x_train_75, x_test_25, y_train_75, y_test_25 = train_test_split(
    train_cleaned, y, test_size=0.25, random_state=1
)


# Define a function to evaluate and store the model's performance
def evaluate_model(model_instance, x_test, y_test, model_name):
    y_pred = model_instance.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    score_test_mse.append(mse)
    score_test_r2.append(r2)
    model.append(model_name)


# Decision Tree Regressor
model_decisiontree_train90 = DecisionTreeRegressor(random_state=0)
model_decisiontree_train90.fit(x_train_90, y_train_90)
evaluate_model(
    model_decisiontree_train90, x_test_10, y_test_10, "model_decisiontree_train90"
)

model_decisiontree_train75 = DecisionTreeRegressor(random_state=0)
model_decisiontree_train75.fit(x_train_75, y_train_75)
evaluate_model(
    model_decisiontree_train75, x_test_25, y_test_25, "model_decisiontree_train75"
)

# LASSO
model_lasso_train90 = Lasso(alpha=0.0005)
model_lasso_train90.fit(x_train_90, y_train_90)
evaluate_model(model_lasso_train90, x_test_10, y_test_10, "model_lasso_train90")

model_lasso_train75 = Lasso(alpha=0.0005)
model_lasso_train75.fit(x_train_75, y_train_75)
evaluate_model(model_lasso_train75, x_test_25, y_test_25, "model_lasso_train75")

# KNN
model_knn_train90 = KNeighborsRegressor(n_neighbors=6)
model_knn_train90.fit(x_train_90, y_train_90)
evaluate_model(model_knn_train90, x_test_10, y_test_10, "model_knn_train90")

model_knn_train75 = KNeighborsRegressor(n_neighbors=6)
model_knn_train75.fit(x_train_75, y_train_75)
evaluate_model(model_knn_train75, x_test_25, y_test_25, "model_knn_train75")

# Final Scores
final_scores = pd.DataFrame(
    {
        "model_name": model,
        "score_test_r2": score_test_r2,
        "score_test_mse": score_test_mse,
    }
)
print(final_scores)


# Predictions for each model
def make_predictions(model_instance, test_data, model_name):
    y_predict = model_instance.predict(test_data)
    result = pd.DataFrame()
    result["Id"] = test["Id"]
    result["SalePrice"] = y_predict
    print(f"{model_name} Predictions")
    print(result.head())


# Decision Tree Predictions
make_predictions(model_decisiontree_train90, test_cleaned, "Decision Tree")

# LASSO Predictions
make_predictions(model_lasso_train90, test_cleaned, "LASSO")

# KNN Predictions
make_predictions(model_knn_train75, test_cleaned, "KNN")
