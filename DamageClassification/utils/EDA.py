import pickle
import pandas as pd

# the following two line are only for MacOS to pass one exception
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns


# check the data in the csv format
merged_csv_path = '../data/data_all.csv'
data_raw = pd.read_csv(merged_csv_path)
print("Number of rows in data =", data_raw.shape[0])
print("Number of columns in data =", data_raw.shape[1])
print("\n")
print("**Sample data:**")
print(data_raw.head(5))
print("\n")

# check the data in the pickle format
pickel_path = '../data/data_all.pickel'
with open(pickel_path, 'rb') as handle:
    data = pickle.load(handle)

damages = data['damages']
features = data['features']

d_nums = damages.sum(axis=0)
f_nums = features.sum(axis=0)

# check whether all types of damage and feature appear or not
print('Do all types of damage appear? ', all(d_nums))
print('Do all types of feature appear? ', all(f_nums))

# plot the quantity of each damage
sns.set(font_scale=2)
plt.figure(figsize=(15,8))
ax = sns.barplot(['1', '2', '3', '4', '5'], d_nums)
plt.title("The quantity of each damage", fontsize=24)
plt.ylabel('Number of damages', fontsize=18)
plt.xlabel('Damage Type ', fontsize=18)
plt.show()

# plot the quantity of each feature
sns.set(font_scale=2)
plt.figure(figsize=(15,8))
ax2 = sns.barplot([x for x in range(500)], f_nums)
plt.title("The quantity of each feature", fontsize=24)
plt.ylabel('Number of features', fontsize=18)
plt.xlabel('Feature Type ', fontsize=18)
plt.show()