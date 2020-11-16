import requests
import numpy as np
import json
import os.path
import time
from helper_functions import translate_coordinate, identify_centroid, calculate_circle_radius, within_boundary_area, segment_address, divide_bounding_box, remove_duplicate, extract_date
from shapely.geometry import Point
import geopandas as gpd


def extract_poi(max_lat=None, max_lng=None, min_lat=None, min_lng=None, centre_lat=None, centre_lng=None, l=50.0,
                h=50.0, api_key=None):
    """
    This function extracts all landmarks found within a bounding circle defined within a box (with its edges described
    using the following information: max_lat, max_lng, min_lat, min_long).
    In the case where the edges of the bounding box are not defined, the function also accepts a single latitude,
    longitude pair which will be translated into a l x h area (m^2) bounding box. Finally, the landmarks found within
    a bounding circle fitting into the bounding box will be extracted.
    """
    # Generate bounding box coordinates
    if (max_lat is None) & (max_lng is None) & (min_lat is None) & (min_lng is None) & (centre_lat is None) & (centre_lng is None):
        raise ValueError('Please either provide a bounding box defined by its edges (i.e. maximum latitude, maximum longitude, minimum latitude, minimum longitude) or a single latitude, longitude pair')

    elif (centre_lat is not None) & (centre_lng is not None):
        max_lat, max_lng, min_lat, min_lng = translate_coordinate(centre_lat, centre_lng, l, h)

    elif (max_lat is not None) & (max_lng is not None) & (min_lat is not None) & (min_lng is not None):
        centre_lat, centre_lng = identify_centroid(max_lat=max_lat, max_lng=max_lng, min_lat=min_lat, min_lng=min_lng)

    else:
        pass

    if (max_lat is None) | (max_lng is None) | (min_lat is None) | (min_lng is None) | (centre_lat is None) | (centre_lng is None):
        raise ValueError('Please either provide a bounding box defined by its edges (i.e. maximum latitude, maximum longitude, minimum latitude, minimum longitude) or a single latitude, longitude pair')

    radius = calculate_circle_radius(max_lat, max_lng, centre_lat, centre_lng)

    # Pass query into HERE API
    geocode_url = 'https://places.ls.hereapi.com/places/v1/discover/explore'
    geocode_url += '?apiKey=' + api_key
    geocode_url += '&in=' + str(centre_lat) + ',' + str(centre_lng) + ';r=' + str(radius)
    geocode_url += '&size' + str(9999)
    geocode_url += '&pretty'

    return requests.get(geocode_url).json()


def format_query_result(query_result):
    """
    This function takes in the result of the HERE API and formats it
    into a list of geojson dictionary which will be returned. The list will also be
    saved as a local json file.
    """
    poi_data = []
    for i in range(len(query_result)):
        lat = query_result[i]['position'][0]
        lng = query_result[i]['position'][1]

        if not within_boundary_area(lat, lng, min_lat, max_lat, min_lng, max_lng):
            continue

        if 'tags' in query_result[i].keys():
            tags = query_result[i]['tags'][0]
        else:
            tags = {}

        poi_dict = {
            'type': 'Feature',
            'geometry': {'location': {'type': 'Point',
                                      'coordinates': [lat, lng]}},
            'properties': {'address': segment_address(query_result[i]['vicinity'].replace('<br/>', ', ')),
                           'name': query_result[i]['title'],
                           'place_type': [query_result[i]['category']['id']],
                           'tags': tags,
                           'source': 'HereMap',
                           'requires_verification': {'summary': 'No'}},
            'id': str(query_result[i]['id']),
            'extraction_date': extract_date()
        }

        poi_data.append(poi_dict)

    return poi_data


def pixelise_region(coordinates, shapefile):
    """
    This function filters out a list of coordinates based on whether it intersects with the study area's shapefile.
    """
    return [coordinate for coordinate in coordinates if
            (np.sum(shapefile['geometry'].apply(lambda x: Point(coordinate[1], coordinate[0]).within(x))) != 0) |
            (np.sum(shapefile['geometry'].apply(lambda x: Point(coordinate[3], coordinate[0]).within(x))) != 0) |
            (np.sum(shapefile['geometry'].apply(lambda x: Point(coordinate[1], coordinate[2]).within(x))) != 0) |
            (np.sum(shapefile['geometry'].apply(lambda x: Point(coordinate[3], coordinate[2]).within(x))) != 0)]


if __name__ == '__main__':
    # Insert your own app id and app code. Refer to: https://developer.here.com/documentation/geocoder/common/credentials.html
    api_key = ''
    wait_time = 15  # sets the number of minutes to wait between each query when your API limit is reached
    output_filename = 'heremap_poi.json'

    # The area of interest is defined using the coordinates of a bounding box (i.e. maximum latitude, maximum longitude, minimum latitude, minimum longitude).
    # Changi Business Park
    # max_lat = 1.339397
    # min_lat = 1.331747
    # max_lng = 103.969027
    # min_lng = 103.961258

    # Orchard
    # max_lat = 1.3206
    # min_lat = 1.2897
    # max_lng = 103.8562
    # min_lng = 103.8153

    # Singapore
    max_lat = 1.4758
    min_lat = 1.15
    max_lng = 104.0939
    min_lng = 103.6005

    # Define the dimensions of the query box (assumed to take the shape of a square)
    querybox_dim = 100.0

    # Import shapefile
    shapefile_df = gpd.read_file('master-plan-2014-planning-area-boundary-web/master-plan-2014-planning-area-boundary-web-shp/MP14_PLNG_AREA_WEB_PL.shp')
    shapefile_df = shapefile_df.to_crs(epsg="4326")
    shapefile_df = shapefile_df[shapefile_df['PLN_AREA_N'] == 'TAMPINES'].reset_index(drop=True)

    # Obtain a list of coordinates for each query box
    coordinate_list = divide_bounding_box(max_lat=max_lat, min_lat=min_lat, max_lng=max_lng, min_lng=min_lng,
                                          querybox_dim=querybox_dim)
    print('Number of queries before filtering: {}'.format(len(coordinate_list)))
    coordinate_list = pixelise_region(coordinate_list, shapefile_df)
    print('Number of queries after filtering: {}'.format(len(coordinate_list)))

    # Extract POI information
    i = 1
    for coordinate in coordinate_list:
        print('Processing query {}/{}'.format(i, len(coordinate_list)))

        not_successful = True
        while not_successful:
            try:
                query_result = extract_poi(max_lat=coordinate[2], max_lng=coordinate[3], min_lat=coordinate[0],
                                           min_lng=coordinate[1], api_key=api_key)
                if query_result['results']['items']:
                    if os.path.exists(output_filename):
                        with open(output_filename) as json_file:
                            feature_collection = json.load(json_file)
                            feature_collection['features'] += format_query_result(query_result['results']['items'])

                        with open(output_filename, 'w') as json_file:
                            json.dump(feature_collection, json_file)

                    else:
                        with open(output_filename, 'w') as json_file:
                            feature_collection = {'type': 'FeatureCollection',
                                                  'features': format_query_result(query_result['results']['items'])}
                            json.dump(feature_collection, json_file)

                else:
                    pass

                not_successful = False

            except requests.exceptions.ConnectionError:
                print('Connection Error. Pausing query for {} minutes...'.format(wait_time))
                time.sleep(wait_time * 60)

            except ValueError:
                print('Pausing query for {} minutes...'.format(wait_time))
                time.sleep(wait_time * 60)

        i += 1

    # Removing duplicate data
    with open(output_filename) as json_file:
        feature_collection = json.load(json_file)

    print('Initial number of data points: {}'.format(len(feature_collection['features'])))
    feature_collection['features'] = remove_duplicate(feature_collection['features'])
    print('Final number of data points: {}'.format(len(feature_collection['features'])))

    with open(output_filename, 'w') as json_file:
        json.dump(feature_collection, json_file)
