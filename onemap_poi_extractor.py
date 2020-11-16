import requests
import json
import time
import os
from helper_functions import generate_id, remove_duplicate, extract_date
import pandas as pd
from shapely.geometry import Polygon


def extract_theme(query_name):
    """
    This function extracts all locations relevant to the theme.
    """
    # Pass query into OneMap API
    geocode_url = 'https://developers.onemap.sg/privateapi/themesvc/retrieveTheme'
    geocode_url += '?queryName=' + query_name
    geocode_url += '&token=' + token

    while True:
        try:
            return requests.get(geocode_url).json()

        except json.decoder.JSONDecodeError:
            time.sleep(5)


def extract_address(query_dict):
    formatted_address = ''
    address = {}
    if 'ADDRESSBLOCKHOUSENUMBER' in query_dict.keys():
        address.update({'house_number': query_dict['ADDRESSBLOCKHOUSENUMBER']})
        formatted_address += query_dict['ADDRESSBLOCKHOUSENUMBER'] + ' '

    if 'ADDRESSSTREETNAME' in query_dict.keys():
        address.update({'road': query_dict['ADDRESSSTREETNAME']})
        formatted_address += query_dict['ADDRESSSTREETNAME'] + ' '

    if 'ADDRESSUNITNUMBER' in query_dict.keys():
        address.update({'unit': 'Unit ' + query_dict['ADDRESSUNITNUMBER']})
        formatted_address += 'Unit ' + query_dict['ADDRESSUNITNUMBER'] + ' '

    if 'ADDRESSFLOORNUMBER' in query_dict.keys():
        address.update({'floor_number': 'Level ' + query_dict['ADDRESSFLOORNUMBER']})
        formatted_address += 'Level ' + query_dict['ADDRESSFLOORNUMBER'] + ' '

    if 'ADDRESSPOSTALCODE' in query_dict.keys():
        address.update({'postcode': ', Singapore ' + query_dict['ADDRESSPOSTALCODE']})
        formatted_address += ', Singapore ' + query_dict['ADDRESSPOSTALCODE']

    if formatted_address == '':
        formatted_address = 'Singapore'

    address.update({'formatted_address': formatted_address})

    return address


def extract_bounds(bound_str):
    coordinates = bound_str.split('|')
    bound_coordinates = [(float(latlng.split(',')[1]), float(latlng.split(',')[0])) for latlng in coordinates]
    centroid = Polygon(bound_coordinates).centroid
    return bound_coordinates, centroid.y, centroid.x


def extract_tags(query_dict):
    temp_dict = {}
    if "DESCRIPTION" in query_dict.keys():
        temp_dict.update({'description': query_dict['DESCRIPTION']})

    if 'ADDRESSTYPE' in query_dict.keys():
        temp_dict.update({'address_type': query_dict['ADDRESSTYPE']})

    if 'ADDRESSBUILDINGNAME' in query_dict.keys():
        temp_dict.update({'building_name': query_dict['ADDRESSBUILDINGNAME']})

    return temp_dict


def map_theme_placetype(theme):
    return [theme_mapping[theme_mapping['themes'] == theme]['mapped_placetype'].tolist()[0]]


def format_query_result(query_result, theme):
    """
    This function takes in the result of the OneMap API and formats it
    into a geojson dictionary which will be returned. The dictionary will also be
    saved as a local json file.
    """
    poi_data = []

    if len(query_result) == 0:
        return poi_data

    for i in range(len(query_result)):
        if i == 0:
            print(query_result[i])

        bound_coordinates = None
        if '|' in query_result[i]['LatLng']:
            bound_coordinates, lat, lng = extract_bounds(query_result[i]['LatLng'])
        else:
            lat, lng = [float(item) for item in query_result[i]['LatLng'].split(',')]

        onemap_address = extract_address(query_result[i])

        poi_dict = {
            'type': 'Feature',
            'geometry': {'location': {'type': 'Point',
                                      'coordinates': [lat, lng]}},
            'properties': {'address': onemap_address,
                           'name': query_result[i]['NAME'],
                           'place_type': map_theme_placetype(theme),
                           'tags': extract_tags(query_result[i]),
                           'source': 'OneMap',
                           'requires_verification': {'summary': 'No'}}
        }

        if bound_coordinates is not None:
            poi_dict['geometry'].update({'bound': {'type': 'Polygon',
                                                   'coordinates': bound_coordinates}})

        poi_dict['id'] = str(generate_id(poi_dict))
        poi_dict['extraction_date'] = extract_date()

        if i == 0:
            print(poi_dict)
            print()

        poi_data.append(poi_dict)

    return poi_data


def extract_query_name(themes):
    geocode_url = 'https://developers.onemap.sg/privateapi/themesvc/getAllThemesInfo'
    geocode_url += '?token=' + str(token)

    while True:
        try:
            query_theme = [(theme_dict['THEMENAME'], theme_dict['QUERYNAME']) for theme_dict in
                           requests.get(geocode_url).json()['Theme_Names'] if theme_dict['THEMENAME'] in themes]
            theme_tuple, query_tuple = zip(*query_theme)
            return list(theme_tuple), list(query_tuple)

        except requests.exceptions.ConnectionError:
            print('Connection Error. Pausing query for {} minutes...'.format(wait_time))
            time.sleep(wait_time)


if __name__ == '__main__':
    # Insert your own app id and app code.
    token = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOjMyMTYsInVzZXJfaWQiOjMyMTYsImVtYWlsIjoiaWFtcmF5bW9uZGxvd0BnbWFpbC5jb20iLCJmb3JldmVyIjpmYWxzZSwiaXNzIjoiaHR0cDpcL1wvb20yLmRmZS5vbmVtYXAuc2dcL2FwaVwvdjJcL3VzZXJcL3Nlc3Npb24iLCJpYXQiOjE1OTkwMTY4OTEsImV4cCI6MTU5OTQ0ODg5MSwibmJmIjoxNTk5MDE2ODkxLCJqdGkiOiJkYmU5ZDY5MWU5YzIzZTU1NjFkNDU3MWUzNDcyN2FkYyJ9.8860MU5Dqa3P7pcJ8-j3KUfazqasurgUcHSBRZBEl5M'
    wait_time = 15  # sets the number of minutes to wait between each query when your API limit is reached
    output_filename = 'onemap_poi_themes.json'

    # Extract query name of selected place types
    theme_mapping = pd.read_csv('onemap_theme_mapping.csv')
    themes, query_names = extract_query_name(theme_mapping['themes'].to_list())
    # Extract POI information based on selected place types
    i = 1
    for j in range(len(query_names)):
        print('Extracting {}...{}/{} themes'.format(query_names[j], i, len(query_names)))

        not_successful = True
        while not_successful:
            try:
                query_result = extract_theme(query_names[j])
                not_successful = False

            except requests.exceptions.ConnectionError:
                print('Connection Error. Pausing query for {} minutes...'.format(wait_time))
                time.sleep(wait_time)

        i += 1

        if os.path.exists(output_filename):
            with open(output_filename) as json_file:
                feature_collection = json.load(json_file)
                feature_collection['features'] += format_query_result(query_result['SrchResults'][2:], themes[j])

            with open(output_filename, 'w') as json_file:
                json.dump(feature_collection, json_file)

        else:
            with open(output_filename, 'w') as json_file:
                feature_collection = {'type': 'FeatureCollection',
                                      'features': format_query_result(query_result['SrchResults'][2:], themes[j])}
                json.dump(feature_collection, json_file)

    # Remove duplicated information
    with open(output_filename) as json_file:
        feature_collection = json.load(json_file)

    print('Initial number of data points: {}'.format(len(feature_collection['features'])))
    feature_collection['features'] = remove_duplicate(feature_collection['features'])
    print('Final number of data points: {}'.format(len(feature_collection['features'])))

    with open(output_filename, 'w') as json_file:
        json.dump(feature_collection, json_file)
