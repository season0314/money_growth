# %%
import csv
import pandas as pd
import numpy as np
data = pd.read_csv("data/enterprise_account_health_score.csv")
#print(data.head())

ah_df = data.iloc[:, 2:10]


# %%
## remove rows with all features are missing

# Define a list of columns to check
columns_to_check = [ 'avg_active_to_paid_rate',
       'avg_active_user_percent_change', 'max_sophistication',
       'avg_active_atl_users', 'activity_workflow_consult',
       'activity_training', 'activity_br']

 #Check if all specified columns are all missing (contain only NaN values) for each row
rows_all_columns_missing = ah_df[columns_to_check].isna().all(axis=1)

# Print the result
print(rows_all_columns_missing)
sum(rows_all_columns_missing)

ah_df = ah_df[-rows_all_columns_missing]
ah_df.shape
# %%
target = ah_df["arr_growth_rate"]
feature = ah_df.loc[:, ah_df.columns != "arr_growth_rate"]


# %%
feature.isna()



# Calculate the median for the entire DataFrame
median_values = feature.median()

# Replace NaN values in the entire DataFrame with the corresponding median values
feature = feature.fillna(median_values)


# %%

print(feature.shape)
print(ah_df.columns)
# %%
type(target)
# %%
target = target.values
feature = feature.to_numpy()
# %%
target.shape
feature.shape
# %%
np.save("data/saved", np.array({"target" : target, 
                                   "feature" : feature}))
# %%

print(feature)
# %%
target.shape
# %%
