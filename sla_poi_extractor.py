import json
import pandas as pd
import geopandas as gpd
from pyproj import Proj


def perform_abbreviation_mapping(gpd_file):
    """
    Maps the terms in the TRADE_CODE, TRADE_TYPE and DATA_TYPE fields into
    its human readable form.
    """
    sla_abbreviation = pd.read_csv('sla_abbreviation.csv')
    abbreviation_list = sla_abbreviation['trade_code'].tolist()

    # Perform mapping for TRADE_CODE
    for trade_code in trade_codes:
        tradecode_index = abbreviation_list.index(trade_code)
        index_list = gpd_file.index[gpd_file['TRADE_CODE'] == trade_code].tolist()
        gpd_file.loc[index_list, 'TRADE_CODE'] = sla_abbreviation.loc[tradecode_index, 'new_description']

    # Perform mapping for TRADE_TYPE
    for trade_type in trade_types:
        tradetype_index = abbreviation_list.index(trade_type)
        index_list = gpd_file.index[gpd_file['TRADE_TYPE'] == trade_type].tolist()
        gpd_file.loc[index_list, 'TRADE_TYPE'] = sla_abbreviation.loc[tradetype_index, 'new_description']

    # Perform mapping for DATA_TYPE
    for data_type in data_types:
        datatype_index = abbreviation_list.index(data_type)
        index_list = gpd_file.index[gpd_file['DATA_TYPE_'] == data_type].tolist()
        gpd_file.loc[index_list, 'DATA_TYPE_'] = sla_abbreviation.loc[datatype_index, 'new_description']

    return gpd_file


def capitalise_string(string):
    """
    Capitalise the first letter of each word in a string.
    """
    capitalised_string = ''
    string_list = string.lower().split(' ')
    for i in range(len(string_list)):
        if not string_list[i]:
            continue
        elif string_list[i][0] != '(':
            capitalised_string += string_list[i].capitalize() + ' '
        else:
            capitalised_string += '(' + string_list[i][1:].capitalize() + ' '

    return capitalised_string[:-1]


def format_address(gpd_row):
    """
    Assign the address information into its respective fields in a dictionary format.
    """
    formatted_address = ''
    address = {}

    if pd.notna(gpd_row['HOUSE_BLK_']):
        address['house_number'] = str(gpd_row['HOUSE_BLK_'])
        formatted_address += str(gpd_row['HOUSE_BLK_']) + ' '

    if pd.notna(gpd_row['ROAD_NAME']):
        address['road'] = capitalise_string(gpd_row['ROAD_NAME'])
        formatted_address += capitalise_string(gpd_row['ROAD_NAME']) + ' '

    if pd.notna(gpd_row['LEVEL_NO']):
        address['level'] = 'Level {}'.format(gpd_row['LEVEL_NO'])
        formatted_address += 'Level {} '.format(gpd_row['LEVEL_NO'])

    if pd.notna(gpd_row['UNIT_NO']):
        address['unit'] = 'Unit {}'.format(gpd_row['UNIT_NO'])
        formatted_address += 'Unit {} '.format(gpd_row['UNIT_NO'])

    address['city'] = 'Singapore'
    formatted_address += 'Singapore '

    if pd.notna(gpd_row['POSTAL_CD']):
        address['postcode'] = str(gpd_row['POSTAL_CD'])
        formatted_address += str(gpd_row['POSTAL_CD']) + ' '

    address['formatted_address'] = formatted_address[:-1]

    return address


def extract_tags(gpd_row):
    """
    Extract POI tags.
    """
    tags = {}

    if pd.notna(gpd_row['TRADE_BRAN']):
        tags['parent'] = capitalise_string(gpd_row['TRADE_BRAN'])

    if pd.notna(gpd_row['TRADE_TYPE']):
        tags['trade_type'] = gpd_row['TRADE_TYPE']

    if pd.notna(gpd_row['DATA_TYPE_']):
        tags['existence'] = gpd_row['DATA_TYPE_']

    return tags


def format_feature(gpd_file):
    """
    Formats the POI features into GeoJSON format.
    """
    features = []
    for i in range(len(gpd_file)):
        print('Processing feature {}/{}'.format(i + 1, len(gpd_file)))
        # print(gpd_file.loc[i, :])
        # print()

        poi_dict = {
            'type': 'Feature',
            'geometry': {'location': {'type': 'Point',
                                      'coordinates': [gpd_file.loc[i, 'LAT'], gpd_file.loc[i, 'LNG']]}},
            'properties': {'address': format_address(gpd_file.loc[i, :]),
                           'name': capitalise_string(gpd_file.loc[i, 'TRADE_NAME']),
                           'place_type': [gpd_file.loc[i, 'TRADE_CODE']],
                           'tags': extract_tags(gpd_file.loc[i, :]),
                           'source': 'SLA',
                           'requires_verification': {'summary': 'No'}},
            'id': str(gpd_file.loc[i, 'OBJECTID']),
            'extraction_date': '20170421'
        }

        features.append(poi_dict)

        # print(poi_dict)
        # print()

    return features


def remove_duplicate(poi_data):
    dropped_index = []
    id_list = []

    for i in range(len(poi_data)):
        if poi_data[i]['id'] in id_list:
            dropped_index.append(i)
        else:
            id_list.append(poi_data[i]['id'])

    for index in sorted(dropped_index, reverse=True):
        del poi_data[index]

    return poi_data


if __name__ == '__main__':
    # Import Shape files
    trade_codes = ['9ANSCH', '9ATM', '9CC', '9CCARE', '9CDEF', '9CENI', '9CHNTE', '9CHU', '9CLNI', '9COT',
                   '9FF', '9FLSCH', '9FSSCH', '9GNS', '9HDBBT', '9HEC', '9HOSP', '9HOSPI', '9HOT',
                   '9INDTE', '9INSEC', '9ITE', '9JC', '9KG', '9LBH', '9LIB', '9MOS', '9NPC', '9OTHIN',
                   '9PBCOM', '9PINT', '9PO', '9POL', '9POLY', '9PRI', '9PTL', '9RCLUB', '9RESCH',
                   '9SCARE', '9SCTRE', '9SEC', '9SHTEM', '9SPSCH', '9SPT', '9SWC', '9SYNA', '9TCH',
                   '9TI', '9VET', '9VI', '19BDPT', '19BINT', '19BOMB', '19BTER']

    trade_types = ['H', 'B']

    data_types = ['EXTG', 'UC', 'PROP']

    for trade_code in trade_codes:
        if trade_code == '9ANSCH':
            gpd_file = gpd.read_file(
                'DS27 - SLA Street Directory Premium/SDP-2016-0006/POI_042017/{}.shp'.format(trade_code))
        else:
            gpd_file = pd.concat([gpd_file, gpd.read_file(
                'DS27 - SLA Street Directory Premium/SDP-2016-0006/POI_042017/{}.shp'.format(trade_code))],
                                 ignore_index=True)

    # Perform abbreviation mapping
    gpd_file = perform_abbreviation_mapping(gpd_file)

    # Extract latitude longitude information
    proj = Proj(gpd_file.crs)
    lng_lat = [proj(geometry.x, geometry.y, inverse=True) for geometry in gpd_file['geometry']]
    lng, lat = zip(*lng_lat)
    gpd_file['LAT'] = lat
    gpd_file['LNG'] = lng

    # Transform into GeoJSON format
    output_filename = 'sla_poi.json'

    with open(output_filename, 'w') as json_file:
        feature_collection = {'type': 'FeatureCollection',
                              'features': format_feature(gpd_file)}
        json.dump(feature_collection, json_file)

    # Remove duplicated information
    with open(output_filename) as json_file:
        feature_collection = json.load(json_file)

    print('Initial number of data points: {}'.format(len(feature_collection['features'])))
    feature_collection['features'] = remove_duplicate(feature_collection['features'])
    print('Final number of data points: {}'.format(len(feature_collection['features'])))

    with open(output_filename, 'w') as json_file:
        json.dump(feature_collection, json_file)
