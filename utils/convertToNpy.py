import geopandas as gpd
import numpy as np
import os
import pandas as pd

geojson_files = os.listdir("./raw_dataset")

geojson_path = list(map(lambda x: os.path.join(
    os.getcwd(), "raw_dataset", x), geojson_files))

gdf_list = []

for i in range(len(geojson_path)-1):
    try:
        gdf = gpd.read_file(geojson_path[i])
        gdf['tree_species'] = geojson_files[i][:-13]
        gdf_list.append(gdf)
    except:
        print(geojson_path[i] + " is not a valid geojson file.")

combined = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))

combined.drop(["geometry"], axis=1, inplace=True)
combined.dropna(inplace=True)

set_of_cols = ["B11", "B12", "B2", "B3", "B4",
               "B5", "B6", "B7", "B8", "B8A", "tree_species"]
set_of_cols_summer = list(
    map(lambda x: x + "_1" if x != "tree_species" else x, set_of_cols))
set_of_cols_autumn = list(
    map(lambda x: x + "_2" if x != "tree_species" else x, set_of_cols))

spring_data = combined[set_of_cols]

summer_data = combined[set_of_cols_summer]

autumn_data = combined[set_of_cols_autumn]

summer_column_mapping = {old_col: new_col for old_col,
                         new_col in zip(summer_data, spring_data)}
autumn_column_mapping = {old_col: new_col for old_col,
                         new_col in zip(autumn_data, spring_data)}

summer_data = summer_data.rename(columns=summer_column_mapping)
autumn_data = autumn_data.rename(columns=autumn_column_mapping)

split_seasons = pd.concat(
    [spring_data, summer_data, autumn_data], ignore_index=True)

group_count = split_seasons.groupby(['tree_species']).size()

group_count.sort_values(ascending=False, inplace=True)

filtered_group_count = group_count.loc[lambda x: x > 1200]

print(filtered_group_count)

filter = list(filtered_group_count.index)

print(filter)

filtered_dataset = split_seasons[split_seasons['tree_species'].isin(filter)]

print(filtered_dataset)

data_array = filtered_dataset.to_numpy()

data_csv = filtered_dataset.to_excel("test.xlsx")

np.save('assets/dataset.npy', data_array)
