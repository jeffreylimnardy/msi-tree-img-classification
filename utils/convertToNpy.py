import geopandas as gpd
import numpy as np
import os
import pandas as pd

# read raw geojson files and combine them into a geodataframe with geopandas
geojson_files = os.listdir("./raw_dataset")

geojson_path = list(map(lambda x: os.path.join(
    os.getcwd(), "raw_dataset", x), geojson_files))

gdf_list = []

for i in range(len(geojson_path)):
    try:
        gdf = gpd.read_file(geojson_path[i])
        gdf['tree_species'] = geojson_files[i][:-13]
        print(gdf)
        gdf_list.append(gdf)
    except:
        print(geojson_path[i] + " is not a valid geojson file.")
        os.remove(geojson_path[i])

combined = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))

group_count = combined.groupby(['tree_species']).size()

group_count.sort_values(ascending=False, inplace=True)

print(group_count)


# drop geometry column since empty and not useful
combined.drop(["geometry"], axis=1, inplace=True)

# drop na to clean dataset
combined.dropna(inplace=True)

print(combined.dtypes)

print(combined)

group_count = combined.groupby(['tree_species']).size()

group_count.sort_values(ascending=False, inplace=True)

print(group_count)


data_array = combined.to_numpy()

data_csv = combined.to_excel("test.xlsx")

np.save('assets/dataset_30bands.npy', data_array)
