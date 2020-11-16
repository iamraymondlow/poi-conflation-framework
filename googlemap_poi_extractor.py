import requests
import json
import os.path
import time
import numpy as np
from helper_functions import translate_coordinate, identify_centroid, within_boundary_area, segment_address, divide_bounding_box, remove_duplicate, extract_date, calculate_circle_radius
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime


def format_coordinates(max_lat, max_lng, min_lat, min_lng, centre_lat, centre_lng):
    """
    This function takes in the coordinates of the bounding box and its centroids to provide a string of the following format: 'centre_lat, centre_long&bounds=min_lat, min_lng|max_lat, max_lng'
    """
    centre_coordinate = str(centre_lat) + ', ' + str(centre_lng)
    bottom_left_coordinate = str(min_lat) + ', ' + str(min_lng)
    top_right_coordinate = str(max_lat) + ', ' + str(max_lng)

    return centre_coordinate + '&bounds=' + bottom_left_coordinate + '|' + top_right_coordinate


def extract_placename(query_result):
    if 'name' in query_result.keys():
        return query_result['name']
    else:
        return np.float('nan')


def extract_poi(max_lat=None, max_lng=None, min_lat=None, min_lng=None, lat=None, lng=None, l=50.0, h=50.0,
                api_key=None):
    """
    This function extracts all POI information found within a bounding box defined by its edges
    (i.e. max_lat, max_lng, min_lat, min_lng).
    In the case where the edges of the bounding box are not defined, the function also accepts a
    single latitude, longitude pair which will be translated into a l x h area (m^2) bounding
    box.
    """

    # Generate bounding box coordinates
    if (max_lat is None) & (max_lng is None) & (min_lat is None) & (min_lng is None) & (lat is None) & (lng is None):
        raise ValueError(
            'Please either provide a bounding box defined by its edges (i.e. maximum latitude, maximum longitude, minimum latitude, minimum longitude) or a single latitude, longitude pair')
    elif (lat is not None) & (lng is not None):
        max_lat, max_lng, min_lat, min_lng = translate_coordinate(lat, lng, l, h)
        # print('Type of Information Provided: Centroids')
    elif (max_lat is not None) & (max_lng is not None) & (min_lat is not None) & (min_lng is not None):
        lat, lng = identify_centroid(max_lat=max_lat, max_lng=max_lng, min_lat=min_lat, min_lng=min_lng)
        # print('Type of Information Provied: Bounding Box')
    else:
        pass

    if (max_lat is None) | (max_lng is None) | (min_lat is None) | (min_lng is None) | (lat is None) | (lng is None):
        raise ValueError(
            'Please either provide a bounding box defined by its edges (i.e. maximum latitude, maximum longitude, minimum latitude, minimum longitude) or a single latitude, longitude pair')

    radius = calculate_circle_radius(max_lat, max_lng, lat, lng)

    # Pass query into GOOGLE MAPS API
    geocode_url = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json'
    params = dict(key=api_key,
                  location=str(lat)+','+str(lng),
                  radius=str(radius))

    query_result = requests.get(url=geocode_url, params=params)

    return query_result.json()


# def format_openinghours(openinghour_dict):
#     """
#     Formats the opening hours information into a list of time dictionaries.
#     """
#     opening_hours = []
#     dayofweek = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
#     print(openinghour_dict)
#     print()
#
#     if len(openinghour_dict['periods']) == 1:  # establishment is opened 24hours
#         for day in dayofweek:
#             opening_hours.append({'day': day,
#                                   'open': '0000',
#                                   'close': '0000'})
#
#     else:
#         for i in range(len(openinghour_dict['periods'])):
#             day = openinghour_dict['periods'][i]['open']['day']
#             open_time = openinghour_dict['periods'][i]['open']['time']
#             close_time = openinghour_dict['periods'][i]['close']['time']
#
#             if day == 0:  # Sunday
#                 opening_hours.append({'day': 'Sun',
#                                       'open': open_time,
#                                       'close': close_time})
#             else:  # other days except Sunday
#                 opening_hours.append({'day': dayofweek[day - 1],
#                                       'open': open_time,
#                                       'close': close_time})
#
#         print('before sorting')
#         print(opening_hours)
#         print()
#         opening_hours = [day_dict for day in dayofweek for day_dict in opening_hours if day_dict['day'] == day]
#
#     print(opening_hours)
#     print()
#     return opening_hours
#
#
# def extract_openinghours(place_id, api_key):
#     """
#     Extracts the opening hours information of a POI using its unique place ID
#     """
#     not_successful = True
#     while not_successful:
#         query_url = 'https://maps.googleapis.com/maps/api/place/details/json?place_id={0}&fields=opening_hours&key={1}'.format(
#             place_id, api_key)
#
#         try:
#             query_result = requests.get(query_url).json()
#
#             if query_result['status'] == 'OK' or query_result['status'] == 'NOT_FOUND':
#                 not_successful = False
#                 # time.sleep(5)
#             else:
#                 print(query_result['status'])
#                 print('Pausing query for {} minutes...'.format(wait_time))
#                 time.sleep(wait_time * 60)
#
#         except requests.exceptions.ConnectionError:
#             print('Connection Error. Pausing query for {} minutes...'.format(wait_time))
#             time.sleep(wait_time * 60)
#
#     if 'result' in query_result and 'opening_hours' in query_result['result']:
#         return format_openinghours(query_result['result']['opening_hours'])
#     else:
#         return []


def format_query_result(query_result):
    """
    This function takes in the result of the Google Map API and formats it
    into a list of dictionary which will be returned. The list will also be
    saved as a local json file.
    """
    poi_data = []
    for i in range(len(query_result)):

        lat = query_result[i]['geometry']['location']['lat']
        lng = query_result[i]['geometry']['location']['lng']

        if not within_boundary_area(lat, lng, min_lat, max_lat, min_lng, max_lng):
            # print('Outside of bounding box. Skipping...')
            continue

        if 'bounds' in query_result[i]['geometry'].keys():
            ne_lat = query_result[i]['geometry']['bounds']['northeast']['lat']
            ne_lng = query_result[i]['geometry']['bounds']['northeast']['lng']
            sw_lat = query_result[i]['geometry']['bounds']['southwest']['lat']
            sw_lng = query_result[i]['geometry']['bounds']['southwest']['lng']
            geometry = {'location': {'type': 'Point',
                                     'coordinates': [lat, lng]},
                        'bound': {'type': 'Polygon',
                                  'coordinates': [[sw_lat, ne_lng],
                                                  [ne_lat, ne_lng],
                                                  [ne_lat, sw_lng],
                                                  [sw_lat, sw_lng]]}}
        else:
            geometry = {'location': {'type': 'Point',
                                     'coordinates': [lat, lng]}}

        if 'vicinity' in query_result[i].keys():
            address = segment_address(query_result[i]['vicinity'])
        else:
            address = np.float('nan')

        poi_dict = {
            'type': 'Feature',
            'geometry': geometry,
            'properties': {'address': address,
                           'name': extract_placename(query_result[i]),
                           'place_type': query_result[i]['types'],
                           'source': 'GoogleMap',
                           'requires_verification': {'summary': 'No'}},
            'id': str(query_result[i]['place_id']),
            'extraction_date': extract_date()
        }

        poi_data.append(poi_dict)

    return poi_data


def pixelise_region(coordinates, shapefile):
    """
    This function filters out a list of coordinates based on whether it intersects with the regions stored within
    the shapefile.
    """
    return [coordinate for coordinate in coordinates if
            (np.sum(shapefile['geometry'].apply(lambda x: Point(coordinate[1], coordinate[0]).within(x))) != 0) |
            (np.sum(shapefile['geometry'].apply(lambda x: Point(coordinate[3], coordinate[0]).within(x))) != 0) |
            (np.sum(shapefile['geometry'].apply(lambda x: Point(coordinate[1], coordinate[2]).within(x))) != 0) |
            (np.sum(shapefile['geometry'].apply(lambda x: Point(coordinate[3], coordinate[2]).within(x))) != 0)]


def variable_bounding_box(max_lat, min_lat, max_lng, min_lng, querybox_dim, shapefile, print_progress=True):
    global num_queries

    # Obtain a list of coordinates for the preliminary set of bounding boxes
    coordinate_list = divide_bounding_box(max_lat=max_lat, min_lat=min_lat, max_lng=max_lng, min_lng=min_lng,
                                          querybox_dim=querybox_dim)
    coordinate_list = pixelise_region(coordinate_list, shapefile_tampines)

    # Extract POI information
    i = 1
    for coordinate in coordinate_list:
        if print_progress:
            print('Processing query {}/{}'.format(i, len(coordinate_list)))

        num_queries += 1

        not_successful = True
        while not_successful:
            try:
                query_result = extract_poi(max_lat=coordinate[2], max_lng=coordinate[3], min_lat=coordinate[0],
                                           min_lng=coordinate[1], api_key=api_key)

                if query_result['status'] == 'OK' or query_result['status'] == 'ZERO_RESULTS':
                    not_successful = False
                    time.sleep(1)

                    if query_result['status'] == 'ZERO_RESULTS' or len(query_result['results']) < 20:
                        box_dimensions.append(querybox_dim)
                    elif len(query_result['results']) >= 20 and querybox_dim/2 >= 2.5:
                        variable_bounding_box(coordinate[2], coordinate[0], coordinate[3], coordinate[1],
                                              querybox_dim/2, shapefile, print_progress=False)
                    elif len(query_result['results']) >= 20 and querybox_dim/2 < 2.5:
                        box_dimensions.append(querybox_dim)
                    else:
                        raise ValueError('Number of queries: {} | Query result: {}'.format(len(query_result['results']),
                                                                                          query_result['status']))

                else:
                    print('Pausing query for {} minutes...'.format(wait_time))
                    time.sleep(wait_time * 60)

            except requests.exceptions.ConnectionError:
                print('Connection Error. Pausing query for {} minutes...'.format(wait_time))
                time.sleep(wait_time * 60)

        i += 1

        if query_result['status'] == 'ZERO_RESULTS':
            continue

        if query_result['results']:
            if os.path.exists(output_filename):
                with open(output_filename) as json_file:
                    feature_collection = json.load(json_file)
                    feature_collection['features'] += format_query_result(query_result['results'])

                with open(output_filename, 'w') as json_file:
                    json.dump(feature_collection, json_file)

            else:
                with open(output_filename, 'w') as json_file:
                    feature_collection = {'type': 'FeatureCollection',
                                          'features': format_query_result(query_result['results'])}
                    json.dump(feature_collection, json_file)

        else:
            continue

    return


if __name__ == '__main__':
    api_key = 'AIzaSyCPLMPZWIzUMm1DSFl3ydeTuBuh1Qw4hXI'  # Insert your own google api key. Refer to: https://developers.google.com/maps/documentation/geocoding/get-api-key
    wait_time = 15  # sets the number of minutes to wait between each query when your API limit is reached
    output_filename = 'googlemap_poi_tampines_vbb.json'

    # Define area of interest
    # Changi Business Park
    # max_lat = 1.339397
    # min_lat = 1.331747
    # max_lng = 103.969027
    # min_lng = 103.961258

    # Singapore
    # max_lat = 1.4758
    # min_lat = 1.15
    # max_lng = 104.0939
    # min_lng = 103.6005

    # Tampines
    max_lat = 1.3892
    min_lat = 1.3268
    max_lng = 103.9918
    min_lng = 103.8903

    # Define the dimensions of the query box (assumed to take the shape of a square)
    # querybox_dim = 250.0
    # querybox_dim = 200.0
    # querybox_dim = 150.0
    querybox_dim = 100.0
    # querybox_dim = 50.0
    # querybox_dim = 25.0

    num_queries = 0

    # Import shapefile
    shapefile_df = gpd.read_file('master-plan-2014-planning-area-boundary-web/master-plan-2014-planning-area-boundary-web-shp/MP14_PLNG_AREA_WEB_PL.shp')
    shapefile_df = shapefile_df.to_crs(epsg="4326")
    shapefile_tampines = shapefile_df[shapefile_df['PLN_AREA_N'] == 'TAMPINES'].reset_index(drop=True)

    # Perform variable bounding box algorithm
    box_dimensions = []
    variable_bounding_box(max_lat, min_lat, max_lng, min_lng, querybox_dim, shapefile_tampines)
    print('Total number of queries made: {}'.format(num_queries))
    # Store dimension information
    with open('box_dim.json', 'w') as json_file:
        dimensions_dict = {'results': box_dimensions}
        json.dump(dimensions_dict, json_file)
    # Plot dimension information using histogram
    plt.hist(box_dimensions, bins=20)
    plt.ylabel('Frequency')
    plt.xlabel('Final Bounding Box Dimensions')
    plt.show()


    # Perform fixed bounding box algorithm
    # Obtain a list of coordinates for the preliminary set of bounding boxes
    # coordinate_list = divide_bounding_box(max_lat=max_lat, min_lat=min_lat, max_lng=max_lng, min_lng=min_lng,
    #                                       querybox_dim=querybox_dim)
    # coordinate_list = pixelise_region(coordinate_list, shapefile_tampines)
    # print('Number of preliminary queries for Tampines: {}'.format(len(coordinate_list)))
    #
    # # Extract POI information
    # i = 1
    # estimated_time = []
    # exceeded_limit = 0
    # for coordinate in coordinate_list:
    #     start_time = datetime.now()
    #     store_time = True
    #     print('Processing query {}/{}'.format(i, len(coordinate_list)))
    #
    #     # if i < 4780:
    #     #     i += 1
    #     #     continue
    #
    #     not_successful = True
    #     while not_successful:
    #         try:
    #             query_result = extract_poi(max_lat=coordinate[2], max_lng=coordinate[3], min_lat=coordinate[0],
    #                                        min_lng=coordinate[1], api_key=api_key)
    #
    #             # print(query_result)
    #
    #             if query_result['status'] == 'OK' or query_result['status'] == 'ZERO_RESULTS':
    #                 not_successful = False
    #                 time.sleep(5)
    #
    #                 if (query_result['status'] == 'OK' and len(query_result['results']) < 20) or (query_result['status'] == 'ZERO_RESULTS'):
    #                     pass
    #                 elif query_result['status'] == 'OK' and len(query_result['results']) >= 20:
    #                     print('Number of results: {}'.format(len(query_result['results'])))
    #                     exceeded_limit += 1
    #                 else:
    #                     raise ValueError('Number of queries: {}| Query result: {}'.format(len(query_result['results']), query_result['status']))
    #
    #             else:
    #                 store_time = False
    #                 print(query_result['status'])
    #                 print('Pausing query for {} minutes...'.format(wait_time))
    #                 time.sleep(wait_time * 60)
    #
    #         except requests.exceptions.ConnectionError:
    #             store_time = False
    #             print('Connection Error. Pausing query for {} minutes...'.format(wait_time))
    #             time.sleep(wait_time * 60)
    #
    #     i += 1
    #
    #     if query_result['status'] == 'ZERO_RESULTS':
    #         continue
    #
    #     if query_result['results']:
    #         if os.path.exists(output_filename):
    #             with open(output_filename) as json_file:
    #                 feature_collection = json.load(json_file)
    #                 feature_collection['features'] += format_query_result(query_result['results'])
    #
    #             with open(output_filename, 'w') as json_file:
    #                 json.dump(feature_collection, json_file)
    #
    #         else:
    #             with open(output_filename, 'w') as json_file:
    #                 feature_collection = {'type': 'FeatureCollection',
    #                                       'features': format_query_result(query_result['results'])}
    #                 json.dump(feature_collection, json_file)
    #
    #     else:
    #         continue

    # print('Percentage of query calls that exceeded limit: {}%'.format(exceeded_limit * 100 / len(coordinate_list)))

    # Remove duplicated information
    with open(output_filename) as json_file:
        feature_collection = json.load(json_file)

    print('Initial number of data points: {}'.format(len(feature_collection['features'])))
    feature_collection['features'] = remove_duplicate(feature_collection['features'])
    print('Final number of data points: {}'.format(len(feature_collection['features'])))

    with open(output_filename, 'w') as json_file:
        json.dump(feature_collection, json_file)

    # Include opening hours information
    # with open(output_filename) as json_file:
    #     feature_collection = json.load(json_file)
    #
    # for i in range(len(feature_collection['features'])):
    #     print('Extracting opening hours information {}/{}'.format(i + 1, len(feature_collection['features'])))
    #     if i < 9715:
    #         i += 1
    #         continue
    #
    #     opening_hours = extract_openinghours(feature_collection['features'][i]['id'], api_key)
    #
    #     if opening_hours:
    #         feature_collection['features'][i]['properties']['opening_hours'] = opening_hours
    #     else:
    #         continue
    #
    # with open(output_filename, 'w') as json_file:
    #     json.dump(feature_collection, json_file)
