import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Data collection
data_train = pd.read_excel(r"./data/train/Data_Train.xlsx")
print(data_train.head())
print(data_train.tail(5))

# 2. Data cleaning/Data Preparation
print(data_train.info())
print(data_train.isnull().sum())

print(data_train["Total_Stops"].isnull())

print(data_train[data_train["Total_Stops"].isnull()])

data_train.dropna(inplace=True)

print(data_train.isnull().sum())

print(data_train.info(memory_usage="deep"))

data_copy = data_train.copy()

print(data_copy.columns)


def change_into_dateTime(col):
    data_copy[col] = pd.to_datetime(data_copy[col])


for feature in ["Dep_Time", "Arrival_Time", "Date_of_Journey"]:
    change_into_dateTime(feature)

print(data_copy.dtypes)

print(data_copy["Date_of_Journey"].dt.day)
print(data_copy["Date_of_Journey"].dt.month)
print(data_copy["Date_of_Journey"].dt.year)

data_copy["Journey_Day"] = data_copy["Date_of_Journey"].dt.day
data_copy["Journey_Month"] = data_copy["Date_of_Journey"].dt.month
data_copy["Journey_Year"] = data_copy["Date_of_Journey"].dt.year

print(data_copy.head(3))


def extract_hour_min(df, col):
    df[col + "_Hour"] = df[col].dt.hour
    df[col + "_Minute"] = df[col].dt.minute
    return df.head(3)


print(extract_hour_min(data_copy, "Dep_Time"))
print(extract_hour_min(data_copy, "Arrival_Time"))

cols_to_drop = ["Arrival_Time", "Dep_Time"]

data_copy.drop(cols_to_drop, axis=1, inplace=True)
print(data_copy.head(3))

# 3. Data Analysis

# TODO: Analyze when will most of the flights take off?

print(data_copy.columns)


def flight_dep_time(x):
    if (x > 4) and (x <= 8):
        return "Early Monring"
    elif (x > 8) and (x <= 12):
        return "Monring"
    elif (x > 12) and (x <= 16):
        return "Noon"
    elif (x > 16) and (x <= 20):
        return "Evening"
    elif (x > 20) and (x <= 24):
        return "Night"
    else:
        return "Late Night"


# Answer
print(
    data_copy["Dep_Time_Hour"]
    .apply(flight_dep_time)
    .value_counts()
    .plot(kind="bar", color="green")
)
plt.show()


# Preprocess duration
def preprocess_duration(x):
    if "h" not in x:
        x = "0h" + " " + x
    elif "m" not in x:
        x = x + " " + "0m"

    return x


data_copy["Duration"] = data_copy["Duration"].apply(preprocess_duration)

data_copy["Duration_hours"] = data_copy["Duration"].apply(
    lambda x: int(x.split(" ")[0][0:-1])
)
data_copy["Duration_mins"] = data_copy["Duration"].apply(
    lambda x: int(x.split(" ")[1][0:-1])
)


data_copy["Duration_total_mins"] = (
    data_copy["Duration"]
    .str.replace("h", "*60")
    .str.replace(" ", "+")
    .str.replace("m", "*1")
    .apply(eval)
)
data_copy["Price"] = data_copy["Price"] * 0.012

sns.lmplot(x="Duration_total_mins", y="Price", data=data_copy)
plt.show()

sns.scatterplot(x="Duration_total_mins", y="Price", data=data_copy)
plt.show()

sns.scatterplot(x="Duration_total_mins", y="Price", hue="Total_Stops", data=data_copy)
plt.show()

data_copy["Airline"] == "Jet Airways"

# Which routes of Jet Airways are most popular?
print(
    data_copy[data_copy["Airline"] == "Jet Airways"]
    .groupby("Route")
    .size()
    .sort_values(ascending=False)
)

# Airlines and Price Analysis
sns.boxplot(
    x="Airline", y="Price", data=data_copy.sort_values("Price", ascending=False)
)
plt.xticks(rotation="vertical")
plt.show()

# 4. Feature engineering
# Feature encoding technique: one-hot encoding
# string to number, vector

cat_col = [col for col in data_copy.columns if data_copy[col].dtype == "object"]

num_col = [col for col in data_copy.columns if data_copy[col].dtype != "object"]

print(data_copy["Source"].unique())

data_copy["Source"].apply(lambda x: 1 if x == "Banglore" else 0)

for sub_category in data_copy["Source"].unique():
    data_copy["Source_" + sub_category] = data_copy["Source"].apply(
        lambda x: 1 if x == sub_category else 0
    )

print(data_copy)

# Optimized feature enconding
print(data_copy["Airline"].unique())

data_copy.groupby(["Airline"])["Price"].mean().sort_values()
airlines = data_copy.groupby(["Airline"])["Price"].mean().sort_values().index

# Create dict
dict_airlines = {key: index for index, key in enumerate(airlines, 0)}

data_copy["Airline"] = data_copy["Airline"].map(dict_airlines)

print(data_copy["Airline"])

print(data_copy["Destination"].unique())

data_copy["Destination"].replace("New Delhi", "Delhi", inplace=True)


dest = data_copy.groupby(["Destination"])["Price"].mean().sort_values().index

# Create dict
dict_dest = {key: index for index, key in enumerate(dest, 0)}

data_copy["Destination"] = data_copy["Destination"].map(dict_dest)

print(data_copy["Destination"])


# Label encoding without sklearn

print(data_copy["Total_Stops"].unique())

stop = {"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}

data_copy["Total_Stops"] = data_copy["Total_Stops"].map(stop)

# Remove data that is not needed

data_copy["Additional_Info"].value_counts()

data_copy.drop(
    columns=[
        "Date_of_Journey",
        "Additional_Info",
        "Duration_total_mins",
        "Source",
        "Journey_Year",
        "Route",
        "Duration",
    ],
    axis=1,
    inplace=True,
)
print(data_copy.columns)


# Outlier detection


def plot(df, col):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    sns.distplot(df[col], ax=ax1)
    sns.boxplot(df[col], ax=ax2, orient="h")
    sns.distplot(df[col], ax=ax3, kde=False)
    plt.show()


print(plot(data_copy, "Price"))

# Replace outliers with the median

q3 = data_copy["Price"].quantile(0.75)
q1 = data_copy["Price"].quantile(0.25)

iqr = q3 - q1
maximum = q3 + 1.5 * iqr
minimum = q1 - 1.5 * iqr

print(maximum)
print(minimum)

print(
    len([price for price in data_copy["Price"] if price > maximum or price < minimum])
)

data_copy["Price"] = np.where(
    data_copy["Price"] >= 420, data_copy["Price"].median(), data_copy["Price"]
)
plot(data_copy, "Price")
