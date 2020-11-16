import pandas as pd
import geopandas as gpd
import numpy as np
import json
from datetime import datetime
import time
from postal.parser import parse_address
from ciso8601 import parse_datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy.fuzz import token_set_ratio
from tqdm import tqdm
import cython
import geopandas as gpd
import os.path
import matplotlib.pyplot as plt
from shapely.geometry import Point, MultiPoint, Polygon
from joblib import load
import glob


def within_bound(lat, lng, shapefile_df):
    #     if (Point(lng, lat).within(shapefile_df.loc[0,'geometry'])):
    #     if (Point(lng, lat).within(shapefile_df.loc[0,'geometry'])) or (Point(lng, lat).within(shapefile_df.loc[1,'geometry'])):
    #     if (Point(lng, lat).within(shapefile_df.loc[0,'geometry'])) or (Point(lng, lat).within(shapefile_df.loc[1,'geometry'])) or (Point(lng, lat).within(shapefile_df.loc[2,'geometry'])):
    #     if (Point(lng, lat).within(shapefile_df.loc[0,'geometry'])) or (Point(lng, lat).within(shapefile_df.loc[1,'geometry'])) or (Point(lng, lat).within(shapefile_df.loc[2,'geometry'])) or (Point(lng, lat).within(shapefile_df.loc[3,'geometry'])):
    if (Point(lng, lat).within(shapefile_df.loc[0, 'geometry'])) or (
    Point(lng, lat).within(shapefile_df.loc[1, 'geometry'])) or (
    Point(lng, lat).within(shapefile_df.loc[2, 'geometry'])) or (
    Point(lng, lat).within(shapefile_df.loc[3, 'geometry'])) or (
    Point(lng, lat).within(shapefile_df.loc[4, 'geometry'])):
        return True
    else:
        return False


def remove_osm_highway(data):
    delete_indices = [i for i in range(len(data)) if 'highway' in data.loc[i, 'properties.place_type']]
    return data.drop(index=delete_indices).reset_index(drop=True)


def remove_nameless_data(data):
    delete_indices = [i for i in range(len(data)) if data.loc[i, 'properties.name'] == '' or
                      data.loc[i, 'properties.name'] == ' ' or
                      pd.isnull(data.loc[i, 'properties.name'])]
    return data.drop(index=delete_indices).reset_index(drop=True)


with open('googlemap_poi_tampines_vbb.json') as json_file:
    googlemap_json = json.load(json_file)

with open('heremap_poi_mapped.json') as json_file:
    heremap_json = json.load(json_file)

with open('osm_poi_v2.json') as json_file:
    osm_json = json.load(json_file)

with open('onemap_poi_themes.json') as json_file:
    onemap_json = json.load(json_file)

with open('sla_poi_mapped.json') as json_file:
    sla_json = json.load(json_file)

# Convert JSON files to dataframes
google_data = pd.json_normalize(googlemap_json['features'])
# print('Google map before: {}'.format(len(google_data)))
# google_data = remove_nameless_data(google_data)
print('Google map after: {}'.format(len(google_data)))

heremap_data = pd.json_normalize(heremap_json['features'])
# print('Here map before: {}'.format(len(heremap_data)))
# heremap_data = remove_nameless_data(heremap_data)
print('Here map after: {}'.format(len(heremap_data)))

osm_data = pd.json_normalize(osm_json['features'])
osm_data = remove_osm_highway(osm_data)
# print('OSM before: {}'.format(len(osm_data)))
# osm_data = remove_nameless_data(osm_data)
print('OSM after: {}'.format(len(osm_data)))

onemap_data = pd.json_normalize(onemap_json['features'])
# print('One map before: {}'.format(len(onemap_data)))
# onemap_data = remove_nameless_data(onemap_data)
print('One map after: {}'.format(len(onemap_data)))

sla_data = pd.json_normalize(sla_json['features'])
# print('SLA before: {}'.format(len(sla_data)))
# sla_data = remove_nameless_data(sla_data)
print('SLA after: {}'.format(len(sla_data)))

combined_data = pd.concat([google_data, heremap_data,
                           osm_data, onemap_data, sla_data], sort=True, ignore_index=True)

# Filtering out POIs that fall outside of Singapore
shapefile_df = gpd.read_file(
    'master-plan-2014-subzone-boundary-no-sea/master-plan-2014-subzone-boundary-no-sea-shp/MP14_SUBZONE_NO_SEA_PL.shp')
shapefile_df = shapefile_df.to_crs(epsg="4326")
tampines_zones = ['TAMPINES NORTH', 'TAMPINES WEST', 'TAMPINES EAST', 'SIMEI', 'XILIN']
shapefile_df = shapefile_df[shapefile_df['SUBZONE_N'].isin(tampines_zones)].reset_index(drop=True)
assert len(shapefile_df) == 5
shapefile_df['area_km2'] = shapefile_df['geometry'].to_crs({'proj': 'cea'}).map(lambda p: p.area / 10 ** 6)

retained_index = [i for i in range(len(combined_data)) if
                  within_bound(combined_data.loc[i, 'geometry.location.coordinates'][0],
                               combined_data.loc[i, 'geometry.location.coordinates'][1],
                               shapefile_df)]
print('Before filtering: {}'.format(len(combined_data)))
combined_data = combined_data.iloc[retained_index].reset_index(drop=True)
print('After filtering: {}'.format(len(combined_data)))

# Retain columns that will be used for identifying duplicates
retained_columns = ['geometry.location.coordinates', 'properties.address.formatted_address',
                    'properties.name', 'id', 'properties.source']
truncated_data = combined_data[retained_columns]
truncated_data[['lat', 'lng']] = pd.DataFrame(truncated_data['geometry.location.coordinates'].values.tolist(),
                                              index=truncated_data.index)

## Ensure that each POI is unique
assert len(truncated_data) == len(truncated_data['id'].unique())


def translate_coordinate(double lat, double


lng, double
l, double
h):
"""
This function takes in a latitude,longitude pair and translates it to
produce a l x h area (m^2) with the original latitude,longitude pair
located at the centroid.
"""
cdef:
double
earth_radius
double
dn
double
de
double
dLat
double
dLng
double
max_lat
double
max_lng
double
min_lat
double
min_lng

# Define the Earth’s radius, assuming a spherical surface
earth_radius = 6378137.0

# Translate location in meters
dn = h / 2.0
de = l / 2.0

# Coordinate translation in radians
dLat = dn / earth_radius
dLng = de / (earth_radius * np.cos(np.pi * lat / 180.0))

# Translate position in decimal degrees
max_lat = lat + dLat * 180.0 / np.pi
max_lng = lng + dLng * 180.0 / np.pi
min_lat = lat - dLat * 180.0 / np.pi
min_lng = lng - dLng * 180.0 / np.pi

return max_lat, max_lng, min_lat, min_lng


def identify_neighbours(data, list coordinates

):
cdef:
double
search_dim
list
neighbour_index
double
max_lat
double
max_lng
double
min_lat
double
min_lng

search_dim = 100.0
max_lat, max_lng, min_lat, min_lng = translate_coordinate(coordinates[0], coordinates[1],
                                                          search_dim, search_dim)

neighbour_index = list(data[(data['lat'] > min_lat) &
                            (data['lat'] < max_lat) &
                            (data['lng'] > min_lng) &
                            (data['lng'] < max_lng)].index)
return neighbour_index


# def identify_duplicates_tfidf_ml(data, centroid_series, int centroid_index, address_matrix, name_matrix, list ml_models):
#     """
#     Identifies duplicated POI points by first identifying the neighbouring POIs and then passing
#     the name similarity and address similarity values into machine learning models to identify duplicated POI.
#     """
#     cdef:
#         list neighbour_index
#         list duplicates_index
#         list temp_index

#     # Step 1: Identify neighbours
#     neighbour_index = identify_neighbours(data, centroid_series['geometry.location.coordinates'])
#     neighbour_index.remove(centroid_index)

#     # Step 2: Identify duplicates among neighbours by passing the name similarity and address similarity
#     # values into the machine learning models.
#     if len(neighbour_index) > 0:  # presence of neighbouring POIs
#         address_vector = address_matrix[centroid_index,:]
#         address_similarity = cosine_similarity(address_matrix[neighbour_index,:], address_vector).reshape(-1,1)
#         name_vector = name_matrix[centroid_index,:]
#         name_similarity = cosine_similarity(name_matrix[neighbour_index,:], name_vector).reshape(-1,1)

#         # Pass name and address similarity values into ML models
#         predict_prob = np.zeros((len(neighbour_index), 2))
#         for model in ml_models:
#             predict_prob += model.predict_proba(np.hstack((address_similarity, name_similarity)))
#         temp_index = list(np.where(np.argmax(predict_prob, axis=1) == 1)[0])
#         duplicate_index = [neighbour_index[i] for i in temp_index]

#         return data.loc[duplicate_index, 'id'].tolist()

#     else:  # no neighbours
#         return []


def identify_duplicates_newtfidf_ml(data, centroid_series, int centroid_index, list


ml_models):
"""
Identifies duplicated POI points by first identifying the neighbouring POIs and then passing
the name similarity and address similarity values into machine learning models to identify duplicated POI.
"""
cdef:
list
neighbour_index
list
duplicates_index
list
temp_index

# Step 1: Identify neighbours
neighbour_index = identify_neighbours(data, centroid_series['geometry.location.coordinates'])
neighbour_index.remove(centroid_index)
neighbour_index.append(centroid_index)

# Step 2: Identify duplicates among neighbours by passing the name similarity and address similarity
# values into the machine learning models.
if len(neighbour_index) > 1:  # presence of neighbouring POIs
    address_corpus = data.loc[neighbour_index, 'properties.address.formatted_address'].fillna('Singapore').tolist()
    address_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3))
    address_matrix = address_vectorizer.fit_transform(address_corpus)

    name_corpus = data.loc[neighbour_index, 'properties.name'].fillna(' ').tolist()
    name_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3))
    name_matrix = name_vectorizer.fit_transform(name_corpus)

    address_vector = address_matrix[-1, :]
    address_similarity = cosine_similarity(address_matrix[:-1, :], address_vector).reshape(-1, 1)
    name_vector = name_matrix[-1, :]
    name_similarity = cosine_similarity(name_matrix[:-1, :], name_vector).reshape(-1, 1)

    # Pass name and address similarity values into ML models
    predict_prob = np.zeros((len(neighbour_index) - 1, 2))
    for model in ml_models:
        predict_prob += model.predict_proba(np.hstack((address_similarity, name_similarity)))
    temp_index = list(np.where(np.argmax(predict_prob, axis=1) == 1)[0])
    duplicate_index = [neighbour_index[i] for i in temp_index if
                       data.loc[neighbour_index[i], 'properties.source'] != centroid_series['properties.source']]

    return data.loc[duplicate_index, 'id'].tolist()

else:  # no neighbours
    return []


def identify_duplicates_string_ml(data, centroid_series, int centroid_index, list


ml_models):
"""
Identifies duplicated POI points by first identifying the neighbouring POIs and then performing
string comparison on the address and name attributes before passing these values into a machine learning
model to identify duplicated POI.
"""
cdef:
list
neighbour_index
list
duplicates_index
list
temp_index
str
centroid_address
str
centroid_name

# Step 1: Identify neighbours
neighbour_index = identify_neighbours(data, centroid_series['geometry.location.coordinates'])
neighbour_index.remove(centroid_index)

# Step 2: Identify duplicates among neighbours by calculating their string similarity score and selecting
# those that have similarity scores that exceed the cutoff score.
if len(neighbour_index) > 0:  # presence of neighbouring POIs
    if pd.isnull(centroid_series['properties.address.formatted_address']):
        centroid_address = ' '
    else:
        centroid_address = centroid_series['properties.address.formatted_address']

    if pd.isnull(centroid_series['properties.name']):
        centroid_name = ' '
    else:
        centroid_name = centroid_series['properties.name']

    neighbour_data = data.iloc[neighbour_index]
    #         neighbour_data['properties.address.formatted_address'] = neighbour_data['properties.address.formatted_address'].fillna(' ')
    #         neighbour_data['properties.name'] = neighbour_data['properties.name'].fillna(' ')

    address_similarity = np.array([token_set_ratio(centroid_address, neighbour_address)
                                   for neighbour_address in
                                   neighbour_data['properties.address.formatted_address'].fillna(
                                       'Singapore').tolist()]).reshape(-1, 1)
    name_similarity = np.array([token_set_ratio(centroid_name, neighbour_name)
                                for neighbour_name in neighbour_data['properties.name'].fillna(' ').tolist()]).reshape(
        -1, 1)

    # Pass name and address similarity values into ML models
    predict_prob = np.zeros((len(neighbour_index), 2))
    for model in ml_models:
        predict_prob += model.predict_proba(np.hstack((address_similarity, name_similarity)))
    temp_index = list(np.where(np.argmax(predict_prob, axis=1) == 1)[0])
    duplicate_index = [neighbour_index[i] for i in temp_index if
                       data.loc[neighbour_index[i], 'properties.source'] != centroid_series['properties.source']]

    return data.loc[duplicate_index, 'id'].tolist()

else:  # no neighbours
    return []


# def identify_duplicates_stringtfidf_ml(data, centroid_series, int centroid_index, address_matrix, list ml_models):
#     """
#     Identifies duplicated POI points by first identifying the neighbouring POIs and then passing
#     the name similarity and address similarity values into machine learning models to identify duplicated POI.
#     """
#     cdef:
#         list neighbour_index
#         list duplicates_index
#         list temp_index

#     # Step 1: Identify neighbours
#     neighbour_index = identify_neighbours(data, centroid_series['geometry.location.coordinates'])
#     neighbour_index.remove(centroid_index)

#     # Step 2: Identify duplicates among neighbours by passing the name similarity and address similarity
#     # values into the machine learning models.
#     if len(neighbour_index) > 0:  # presence of neighbouring POIs
#         address_vector = address_matrix[centroid_index,:]
#         address_similarity = cosine_similarity(address_matrix[neighbour_index,:], address_vector).reshape(-1,1)

#         if pd.isnull(centroid_series['properties.name']):
#             centroid_name = ' '
#         else:
#             centroid_name = centroid_series['properties.name']
#         neighbour_data = data.iloc[neighbour_index]
#         neighbour_data['properties.name'] = neighbour_data['properties.name'].fillna(' ')
#         name_similarity = np.array([token_set_ratio(centroid_name, neighbour_name)
#                                     for neighbour_name in neighbour_data['properties.name'].tolist()]).reshape(-1,1)

#         # Pass name and address similarity values into ML models
#         predict_prob = np.zeros((len(neighbour_index), 2))
#         for model in ml_models:
#             predict_prob += model.predict_proba(np.hstack((address_similarity, name_similarity)))
#         temp_index = list(np.where(np.argmax(predict_prob, axis=1) == 1)[0])
#         duplicate_index = [neighbour_index[i] for i in temp_index]

#         return data.loc[duplicate_index, 'id'].tolist()

#     else:  # no neighbours
#         return []


def identify_duplicates_stringnewtfidf_ml(data, centroid_series, int centroid_index, list


ml_models):
"""
Identifies duplicated POI points by first identifying the neighbouring POIs and then passing
the name similarity and address similarity values into machine learning models to identify duplicated POI.
"""
cdef:
list
neighbour_index
list
duplicates_index
list
temp_index

# Step 1: Identify neighbours
neighbour_index = identify_neighbours(data, centroid_series['geometry.location.coordinates'])
neighbour_index.remove(centroid_index)
neighbour_index.append(centroid_index)

# Step 2: Identify duplicates among neighbours by passing the name similarity and address similarity
# values into the machine learning models.
if len(neighbour_index) > 1:  # presence of neighbouring POIs
    address_corpus = data.loc[neighbour_index, 'properties.address.formatted_address'].fillna('Singapore').tolist()
    address_corpus = ['Singapore' if address == '' else address for address in address_corpus]
    address_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3))
    address_matrix = address_vectorizer.fit_transform(address_corpus)
    address_vector = address_matrix[-1, :]
    address_similarity = cosine_similarity(address_matrix[:-1, :], address_vector).reshape(-1, 1)

    if pd.isnull(centroid_series['properties.name']):
        centroid_name = ' '
    else:
        centroid_name = centroid_series['properties.name']
    neighbour_data = data.iloc[neighbour_index[:-1]]
    name_similarity = np.array([token_set_ratio(centroid_name, neighbour_name)
                                for neighbour_name in neighbour_data['properties.name'].fillna(' ').tolist()]).reshape(
        -1, 1)

    # Pass name and address similarity values into ML models
    predict_prob = np.zeros((len(neighbour_index) - 1, 2))
    for model in ml_models:
        predict_prob += model.predict_proba(np.hstack((address_similarity, name_similarity)))
    temp_index = list(np.where(np.argmax(predict_prob, axis=1) == 1)[0])
    duplicate_index = [neighbour_index[i] for i in temp_index if
                       data.loc[neighbour_index[i], 'properties.source'] != centroid_series['properties.source']]

    return data.loc[duplicate_index, 'id'].tolist()

else:  # no neighbours
    return []


# def identify_duplicates_tfidf(data, centroid_series, int centroid_index, address_matrix, name_matrix, double cutoff_score, double name_weight):
#     """
#     Identifies duplicated POI points by first identifying the neighbouring POIs and then performing
#     TFIDF and calculating cosine similarity on the address and name attributes to identify duplicated POI.
#     """
#     cdef:
#         list neighbour_index
#         list duplicates_index
#         list temp_index

#     # Step 1: Identify neighbours
#     neighbour_index = identify_neighbours(data, centroid_series['geometry.location.coordinates'])
#     neighbour_index.remove(centroid_index)

#     # Step 2: Identify duplicates among neighbours by calculating their cosine similarity score and selecting
#     # those that have similarity scores that exceed the cutoff score.
#     if len(neighbour_index) > 0:  # presence of neighbouring POIs
#         address_vector = address_matrix[centroid_index,:]
#         address_similarity = cosine_similarity(address_matrix[neighbour_index,:], address_vector)
#         name_vector = name_matrix[centroid_index,:]
#         name_similarity = cosine_similarity(name_matrix[neighbour_index,:], name_vector)

#         similarity_score = address_similarity * (1-name_weight) + name_similarity * name_weight
#         temp_index = list(np.where(similarity_score >= cutoff_score)[0])
#         duplicate_index = [neighbour_index[i] for i in temp_index]

#         return data.loc[duplicate_index, 'id'].tolist()

#     else:  # no neighbours
#         return []


def identify_duplicates_newtfidf(data, centroid_series, int centroid_index, double


cutoff_score, double
name_weight):
"""
Identifies duplicated POI points by first identifying the neighbouring POIs and then performing
TFIDF and calculating cosine similarity on the address and name attributes to identify duplicated POI.
"""
cdef:
list
neighbour_index
list
duplicates_index
list
temp_index

# Step 1: Identify neighbours
neighbour_index = identify_neighbours(data, centroid_series['geometry.location.coordinates'])
neighbour_index.remove(centroid_index)
neighbour_index.append(centroid_index)

# Step 2: Identify duplicates among neighbours by calculating their cosine similarity score and selecting
# those that have similarity scores that exceed the cutoff score.
if len(neighbour_index) > 1:  # presence of neighbouring POIs
    address_corpus = data.loc[neighbour_index, 'properties.address.formatted_address'].fillna('Singapore').tolist()
    address_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3))
    address_matrix = address_vectorizer.fit_transform(address_corpus)

    name_corpus = data.loc[neighbour_index, 'properties.name'].fillna(' ').tolist()
    name_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3))
    name_matrix = name_vectorizer.fit_transform(name_corpus)

    address_vector = address_matrix[-1, :]
    address_similarity = cosine_similarity(address_matrix[:-1, :], address_vector)
    name_vector = name_matrix[-1, :]
    name_similarity = cosine_similarity(name_matrix[:-1, :], name_vector)

    similarity_score = address_similarity * (1 - name_weight) + name_similarity * name_weight
    temp_index = list(np.where(similarity_score >= cutoff_score)[0])
    duplicate_index = [neighbour_index[i] for i in temp_index if
                       data.loc[neighbour_index[i], 'properties.source'] != centroid_series['properties.source']]

    return data.loc[duplicate_index, 'id'].tolist()

else:  # no neighbours
    return []


# def identify_duplicates_stringtfidf(data, centroid_series, int centroid_index, address_matrix, double cutoff_score, double name_weight):
#     """
#     Identifies duplicated POI points by first identifying the neighbouring POIs and then performing
#     TFIDF (for address) and string comparison (for name) to identify duplicated POI.
#     """
#     cdef:
#         list neighbour_index
#         list duplicates_index
#         list temp_index

#     # Step 1: Identify neighbours
#     neighbour_index = identify_neighbours(data, centroid_series['geometry.location.coordinates'])
#     neighbour_index.remove(centroid_index)

#     # Step 2: Identify duplicates among neighbours by calculating their cosine similarity score and selecting
#     # those that have similarity scores that exceed the cutoff score.
#     if len(neighbour_index) > 0:  # presence of neighbouring POIs
#         address_vector = address_matrix[centroid_index,:]
#         address_similarity = cosine_similarity(address_matrix[neighbour_index,:], address_vector)

#         if pd.isnull(centroid_series['properties.name']):
#             centroid_name = ' '
#         else:
#             centroid_name = centroid_series['properties.name']
#         name_similarity = np.array([token_set_ratio(centroid_name, neighbour_name)
#                                     for neighbour_name in data.loc[neighbour_index, 'properties.name'].tolist()])

#         similarity_score = address_similarity * (1-name_weight) + name_similarity * name_weight
#         temp_index = list(np.where(similarity_score >= cutoff_score)[0])
#         duplicate_index = [neighbour_index[i] for i in temp_index]

#         return data.loc[duplicate_index, 'id'].tolist()

#     else:  # no neighbours
#         return []


def identify_duplicates_stringnewtfidf(data, centroid_series, int centroid_index, double


cutoff_score, double
name_weight):
"""
Identifies duplicated POI points by first identifying the neighbouring POIs and then performing
TFIDF (for address) and string comparison (for name) to identify duplicated POI.
"""
cdef:
list
neighbour_index
list
duplicates_index
list
temp_index

# Step 1: Identify neighbours
neighbour_index = identify_neighbours(data, centroid_series['geometry.location.coordinates'])
neighbour_index.remove(centroid_index)
neighbour_index.append(centroid_index)

# Step 2: Identify duplicates among neighbours by calculating their cosine similarity score and selecting
# those that have similarity scores that exceed the cutoff score.
if len(neighbour_index) > 1:  # presence of neighbouring POIs
    address_corpus = data.loc[neighbour_index, 'properties.address.formatted_address'].fillna('Singapore').tolist()
    #         address_corpus = ['Singapore' if address == '' else address for address in address_corpus]
    address_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3))
    address_matrix = address_vectorizer.fit_transform(address_corpus)
    address_vector = address_matrix[-1, :]
    address_similarity = cosine_similarity(address_matrix[:-1, :], address_vector).reshape(-1)

    if pd.isnull(centroid_series['properties.name']):
        centroid_name = ' '
    else:
        centroid_name = centroid_series['properties.name']
    name_similarity = np.array([token_set_ratio(centroid_name, neighbour_name)
                                for neighbour_name in
                                data.loc[neighbour_index[:-1], 'properties.name'].tolist()]).reshape(-1)[:-1]

    similarity_score = address_similarity * (1 - name_weight) + (name_similarity / 100.0) * name_weight
    temp_index = list(np.where(similarity_score >= cutoff_score)[0])
    duplicate_index = [neighbour_index[i] for i in temp_index if
                       data.loc[neighbour_index[i], 'properties.source'] != centroid_series['properties.source']]

    return data.loc[duplicate_index, 'id'].tolist()

else:  # no neighbours
    return []


def identify_duplicates_string(data, centroid_series, int centroid_index, cutoff_score, name_weight

):
"""
Identifies duplicated POI points by first identifying the neighbouring POIs and then performing
string comparison on the address and name attributes to identify duplicated POI.
"""
cdef:
list
neighbour_index
list
duplicates_index
list
temp_index
str
centroid_address
str
centroid_name
list
address_similarity
list
name_similarity

# Step 1: Identify neighbours
neighbour_index = identify_neighbours(data, centroid_series['geometry.location.coordinates'])
neighbour_index.remove(centroid_index)

# Step 2: Identify duplicates among neighbours by calculating their string similarity score and selecting
# those that have similarity scores that exceed the cutoff score.
if len(neighbour_index) > 0:  # presence of neighbouring POIs
    if pd.isnull(centroid_series['properties.address.formatted_address']):
        centroid_address = ' '
    else:
        centroid_address = centroid_series['properties.address.formatted_address']

    if pd.isnull(centroid_series['properties.name']):
        centroid_name = ' '
    else:
        centroid_name = centroid_series['properties.name']

    neighbour_data = data.iloc[neighbour_index]
    #         neighbour_data['properties.address.formatted_address'] = neighbour_data['properties.address.formatted_address'].fillna(' ')
    #         neighbour_data['properties.name'] = neighbour_data['properties.name'].fillna(' ')

    address_similarity = [token_set_ratio(centroid_address, neighbour_address)
                          for neighbour_address in
                          neighbour_data['properties.address.formatted_address'].fillna('Singapore').tolist()]
    name_similarity = [token_set_ratio(centroid_name, neighbour_name)
                       for neighbour_name in neighbour_data['properties.name'].fillna(' ').tolist()]

    similarity_score = (np.array(address_similarity) * (1 - name_weight) + np.array(name_similarity) * (
        name_weight)) / 100.0
    temp_index = list(np.where(similarity_score >= cutoff_score)[0])
    duplicate_index = [neighbour_index[i] for i in temp_index if
                       data.loc[neighbour_index[i], 'properties.source'] != centroid_series['properties.source']]

    return data.loc[duplicate_index, 'id'].tolist()

else:  # no neighbours
    return []

# Define cutoff similarity score for identifying duplicates
cutoff_score = 0.85
name_similarity_weight = 0.95

# Vectorise address and name information
# address_corpus = truncated_data['properties.address.formatted_address'].fillna(' ').tolist()
# address_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3))
# address_matrix = address_vectorizer.fit_transform(address_corpus)

# name_corpus = truncated_data['properties.name'].fillna(' ').tolist()
# name_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3))
# name_matrix = name_vectorizer.fit_transform(name_corpus)

# Load machine learning models
model_filenames = glob.glob('gb_model_stringnewtfidf?.joblib')
# model_filenames = glob.glob('gb_model_stringtfidf?.joblib')
ml_models = [load(filename) for filename in model_filenames]

# Identify duplicates using cosine similarity score
duplicate_indices = []
run_time = []
for i in tqdm(range(len(truncated_data))):
    start_time = datetime.now()
#     duplicate_indices.append(identify_duplicates_newtfidf(truncated_data, truncated_data.iloc[i], i,
#                                                           cutoff_score, name_similarity_weight))
#     duplicate_indices.append(identify_duplicates_string(truncated_data, truncated_data.iloc[i],
#                                                         i, cutoff_score, name_similarity_weight))
#     duplicate_indices.append(identify_duplicates_stringnewtfidf(truncated_data, truncated_data.iloc[i], i,
#                                                              cutoff_score, name_similarity_weight))
#     duplicate_indices.append(identify_duplicates_newtfidf_ml(truncated_data, truncated_data.iloc[i], i,
#                                                           ml_models))
#     duplicate_indices.append(identify_duplicates_string_ml(truncated_data, truncated_data.iloc[i],
#                                                            i, ml_models))
    duplicate_indices.append(identify_duplicates_stringnewtfidf_ml(truncated_data, truncated_data.iloc[i],
                                                                   i, ml_models))
    run_time.append(((datetime.now() - start_time)*len(truncated_data))/(60.0 * 60.0))

assert len(combined_data) == len(duplicate_indices)

combined_data['duplicates'] = duplicate_indices
# combined_data.to_csv('singapore_v4.csv'.format(cutoff_score), index=False)
# combined_data.to_csv('singapore_tfidf_{}.csv'.format(cutoff_score), index=False)

print('string comparison approach')
num_zones = 0
print('number of zones considered: {}'.format(num_zones))
print('nuwwmber of POIs considered: {}'.format(len(truncated_data)))
print('area considered: {}'.format(np.sum(shapefile_df.loc[:num_zones,'area_km2'])))
print('maximum time: {}'.format(np.max(run_time)))
print('minimum time: {}'.format(np.min(run_time)))
print('average time: {}'.format(np.mean(run_time)))


def find_all_duplicates(list duplicate_ids, data

):
"""
Searches for the entire list of all identified duplicates and returns their IDs together with
their information.
"""
cdef:
bint
all_duplicates_not_found
list
id_list

all_duplicates_not_found = True

while all_duplicates_not_found:
    filtered_data = data[data['id'].isin(duplicate_ids)].reset_index(drop=True)

    id_list = list(set([item for sublist in filtered_data['duplicates'].tolist() for item in sublist] + duplicate_ids))

    duplicate_ids.sort()
    id_list.sort()
    if id_list == duplicate_ids:
        all_duplicates_not_found = False

    duplicate_ids = id_list

return duplicate_ids, data[data['id'].isin(duplicate_ids)]


def set_keys(poi_dict, keys, value):
    """
    Assign value into a field in poi_dict based on key.
    """
    current = poi_dict
    for i in range(len(keys)):
        key = keys[i]
        if key not in current:  # key currently not found in poi_dict
            if i == (len(keys) - 1):  # last key in dictionary
                current[key] = value
            else:
                current[key] = {}  # add an empty field for the subsequent keys
                current = current[key]
        else:  # key can be found in poi_dict
            if i == (len(keys) - 1):
                raise ValueError('{} is currently occupied'.format(key))
            else:
                current = current[key]

    return poi_dict


def convert_to_geojson(series, str sep = "."

):
"""
Converts the POI series into its GeoJSON format
"""
cdef:
dict
poi_dict
str
index

poi_dict = {}
for index, value in series.iteritems():
    if index == 'duplicates':
        continue
    if (type(value) is np.float64 or type(value) is float) and pd.isnull(value):
        continue
    keys = index.split(sep)
    poi_dict = set_keys(poi_dict, keys, value)

return poi_dict


def segment_address(address):
    cdef:
    dict
    address_dict
    list
    address_list
    str
    val
    str
    key
    str
    capitalised_value
    str
    item
    str
    formatted_address


address_dict = {}
address_list = parse_address(address)
for value, key in address_list:
    capitalised_value = ''
    for item in value.split():
        capitalised_value += item.capitalize() + ' '
    address_dict[key] = capitalised_value[:-1]

formatted_address = ''
for key, value in address_dict.items():
    formatted_address += value + ' '
address_dict['formatted_address'] = formatted_address[:-1]

return address_dict


def merge_location_coordinates(list location_list

):
"""
Finds the centroid from a list of point coordinates.
"""
cdef:
double
lat
double
lng

lat, lng = np.mean(np.array(location_list), axis=0)
return [lat, lng]


def merge_bound_coordinates(list bound_list

):
"""
Finds the largest bound among all possible bounds.
"""
if len(bound_list) <= 1:
    return bound_list
else:
    largest_bound = None
    largest_area = 0.0

    for bound in bound_list:
        bound_area = Polygon(bound).area
        if bound_area > largest_area:
            largest_bound = bound
            largest_area = bound_area

    return largest_bound


#         se_coordinate, _, nw_coordinate, _ = list(zip(*bound_list))
#         l2_distance = np.linalg.norm(np.array(se_coordinate) - np.array(nw_coordinate), axis=1)
#         return bound_list[np.argmax(l2_distance)]


def extract_trusted_source_index(data):
    source_list = data['properties.source'].tolist()

    if 'OneMap' in source_list:
        source_index = [i for i in data.index if data.loc[i, 'properties.source'] == 'OneMap']

    elif 'SLA' in source_list:
        source_index = [i for i in data.index if data.loc[i, 'properties.source'] == 'SLA']

    elif 'GoogleMap' in source_list:
        source_index = [i for i in data.index if data.loc[i, 'properties.source'] == 'GoogleMap']

    elif 'HereMap' in source_list:
        source_index = [i for i in data.index if data.loc[i, 'properties.source'] == 'HereMap']

    elif 'OpenStreetMap' in source_list:
        source_index = [i for i in data.index if data.loc[i, 'properties.source'] == 'OpenStreetMap']

    else:
        raise ValueError(
            'properties.source {} does not fall under one of the five POI source.'.format(data['properties.source']))

    #     print(source_index)
    return source_index


def merge_duplicates(duplicates):
    """
    Merge all of the rows in the dataframe to return a GeoJSON dictionary retaining the most
    amount of information.
    """
    cdef:
    bint
    conflicting_opening_hours
    bint
    multiple_descriptions
    bint
    no_placetype_info
    str
    column
    list
    location_type
    list
    bound_type
    str
    retained_address
    dict
    segmented_address
    list
    names
    list
    place_type
    list
    tags
    list
    descriptions
    list
    opening_hours
    list
    source
    list
    extraction_date
    double
    latest_time


conflicting_opening_hours = False
multiple_descriptions = False
no_placetype_info = False
merged_poi = pd.Series()

trusted_source_index = extract_trusted_source_index(duplicates)
#     print(duplicates[['properties.name','properties.address.formatted_address','properties.source']])
#     print(duplicates.index)
#     print(trusted_source_index)

for column in duplicates.columns:
    #         print(column)
    # feature type
    if column == 'type':
        merged_poi[column] = 'Feature'

    # geometry type
    elif 'geometry' in column and 'type' in column:  # keep geometry type information
        if 'bound' in column:  # bound type
            bound_type = list(set([item for item in duplicates['geometry.bound.type'].tolist() if not pd.isnull(item)]))
            if len(bound_type) == 0:
                continue
            elif len(bound_type) == 1:
                merged_poi['geometry.bound.type'] = bound_type[0]
            else:
                merged_poi['geometry.bound.type'] = 'Polygon'

        elif 'location' in column:  # location type
            location_type = list(
                set([item for item in duplicates['geometry.location.type'].tolist() if not pd.isnull(item)]))
            if len(location_type) == 0:
                continue
            elif len(location_type) == 1:
                merged_poi['geometry.location.type'] = location_type[0]
            else:
                raise ValueError('More than one location type')
        else:
            raise ValueError('Geometry column {} is not considered'.format(column))

    # geometry coordinates
    elif 'geometry' in column and 'coordinates' in column:
        if 'bound' in column:
            merged_bounds = merge_bound_coordinates(
                [item for item in duplicates['geometry.bound.coordinates'].tolist() if type(item) is list])
            if merged_bounds:
                merged_poi[column] = merged_bounds
            else:
                continue

        elif 'location' in column:
            centroid = MultiPoint(
                duplicates.loc[trusted_source_index, 'geometry.location.coordinates'].tolist()).centroid
            merged_poi[column] = [centroid.x, centroid.y]

        else:
            raise ValueError('Geometry column {} is not considered'.format(column))

    # address
    elif 'properties.address' in column:
        if 'formatted_address' in column:  # only the formatted address is retained
            retained_address = max(
                duplicates.loc[trusted_source_index, 'properties.address.formatted_address'].tolist(), key=len)
            segmented_address = segment_address(retained_address)
            for key, value in segmented_address.items():
                merged_poi['properties.address.{}'.format(key)] = value

        else:  # ignore other components of the address information
            continue

    # name
    elif column == 'properties.name':
        merged_poi[column] = max(duplicates.loc[trusted_source_index, 'properties.name']
                                 .fillna(' ').tolist(), key=len)

    # place type
    elif column == 'properties.place_type':  # store all place types in a list
        place_type = list(set([item for sublist in duplicates[column].tolist() for item in sublist]))
        #             print('place type: {}'.format(place_type))
        if len(place_type) == 0:
            no_placetype_info = True
        merged_poi[column] = place_type

    # tags
    elif 'properties.tags' in column:
        tags = list(set([item for item in duplicates[column].tolist() if not pd.isnull(item)]))
        if len(tags) == 0:  # no tag information
            continue
        elif len(tags) == 1:  # single tag found
            merged_poi[column] = tags[0]
        else:  # multiple possible tags. stored as a list
            merged_poi[column] = tags

    # description
    elif column == 'properties.description':
        descriptions = [item for item in duplicates[column].tolist() if not pd.isnull(item)]
        if len(descriptions) == 0:  # no description information
            continue
        elif len(descriptions) == 1:
            merged_poi[column] = descriptions[0]
        else:  # flag out instances where there are more than one possible description
            merged_poi[column] = descriptions
            multiple_descriptions = True

    # opening hours
    elif column == 'properties.opening_hours':
        opening_hours = [item for item in duplicates[column].tolist() if type(item) is list]
        if len(opening_hours) == 0:  # no opening hour information
            continue
        elif len(opening_hours) == 1:
            merged_poi[column] = opening_hours[0]
        else:  # flag out instances where there are more than one possible set of opening hours
            merged_poi[column] = opening_hours
            conflicting_opening_hours = True

    # source
    elif column == 'properties.source':  # store all sources in a list
        source = list(set(duplicates[column].tolist()))
        merged_poi[column] = source

    # requires_verification
    elif 'properties.requires_verification' in column:  # if any duplicated poi requires verification, the merged poi will also require verification
        if 'Yes' in duplicates['properties.requires_verification.summary'].tolist():
            merged_poi['properties.requires_verification.summary'] = 'Yes'
            merged_poi['properties.requires_verification.reasons'] = list(set(
                [item for sublist in duplicates['properties.requires_verification.reasons'].tolist() if
                 not pd.isnull(sublist) for item in sublist]))
        else:
            merged_poi['properties.requires_verification.summary'] = 'No'

    # id information
    elif column == 'id':  # store all ids in a list
        merged_poi[column] = duplicates[column].tolist()

    # extraction date information
    elif column == 'extraction_date':  # the latest date will be chosen
        extraction_date = list(set([item for item in duplicates[column].tolist() if not pd.isnull(item)]))
        if len(extraction_date) == 1:
            merged_poi[column] = extraction_date[0]
        elif len(extraction_date) > 1:
            latest_time = 0.0
            latest_index = None
            for i in range(len(extraction_date)):
                unix_time = time.mktime(parse_datetime(extraction_date[i]).timetuple())
                if unix_time > latest_time:
                    latest_time = unix_time
                    latest_index = i

            merged_poi[column] = extraction_date[latest_index]
        else:
            raise ValueError('Extraction date information is missing.')

    elif column == 'duplicates':  # ignore duplicate field
        continue

    else:
        raise ValueError('{} is not considered!'.format(column))

    if conflicting_opening_hours:
        merged_poi['properties.requires_verification.summary'] = 'Yes'
        if 'properties.requires_verification.reasons' in merged_poi:
            merged_poi['properties.requires_verification.reasons'] += ['Conflicting Opening Hours']
        else:
            merged_poi['properties.requires_verification.reasons'] = ['Conflicting Opening Hours']

    if multiple_descriptions:
        merged_poi['properties.requires_verification.summary'] = 'Yes'
        if 'properties.requires_verification.reasons' in merged_poi:
            merged_poi['properties.requires_verification.reasons'] += ['Multiple Descriptions']
        else:
            merged_poi['properties.requires_verification.reasons'] = ['Multiple Descriptions']

    if no_placetype_info:
        merged_poi['properties.requires_verification.summary'] = 'Yes'
        if 'properties.requires_verification.reasons' in merged_poi:
            merged_poi['properties.requires_verification.reasons'] += ['No Place Type Information']
        else:
            merged_poi['properties.requires_verification.reasons'] = ['No Place Type Information']

#     print(merged_poi)
return convert_to_geojson(merged_poi)

unified_features = []
processed_poi_id = []

for i in tqdm(range(len(combined_data))):
    if type(combined_data.loc[i, 'id']) is list:
        assert len(combined_data.loc[i, 'id']) == 1
        if combined_data.loc[i, 'id'][0] in processed_poi_id:
            continue
        else:
            pass

    elif type(combined_data.loc[i, 'id']) is str:
        if combined_data.loc[i, 'id'] in processed_poi_id:
            continue
        else:
            pass

    else:
        raise ValueError('id is of type {}'.format(type(combined_data.loc[i, 'id'])))

    if combined_data.loc[i, 'id'] in processed_poi_id:  # ignore POIs that has been merged
        continue

    if len(combined_data.loc[i, 'duplicates']) != 0:
        duplicate_ids, duplicates = find_all_duplicates(combined_data.loc[i, 'duplicates'], combined_data)
        unified_features.append(merge_duplicates(duplicates))
        processed_poi_id += duplicate_ids
    else:
        unified_features.append(convert_to_geojson(combined_data.loc[i, :]))
        processed_poi_id.append(combined_data.loc[i, 'id'])

print('Number of data points before removing duplicated data: {}'.format(len(combined_data)))
print('Number of data points after removing duplicated data: {}'.format(len(unified_features)))


output_filename = 'unified_poi_tampines.json'

with open(output_filename, 'w') as json_file:
    feature_collection = {'type': 'FeatureCollection',
                          'features': unified_features}
    json.dump(feature_collection, json_file)
