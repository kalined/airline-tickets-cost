import matplotlib.pyplot as plt
import pandas as pd

# Data collection
data_train = pd.read_excel(r"./data/train/Data_Train.xlsx")
print(data_train.head())
print(data_train.tail(5))

# Data cleaning/Data Preparation
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

# Data Analysis

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
