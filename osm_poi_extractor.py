import requests
import json
import os.path
import time
import numpy as np
from helper_functions import segment_address, remove_duplicate, extract_date
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import Point


def query_address(lat, lng):
    # Pass query into HERE API
    search_radius = '10'

    # geocode_url = 'https://places.cit.api.here.com/places/v1/discover/explore'
    geocode_url = 'https://places.ls.hereapi.com/places/v1/discover/explore'
    geocode_url += '?apiKey=' + here_app_key
    geocode_url += '&mode=retrieveAll'
    geocode_url += '&in=' + str(lat) + ',' + str(lng) + ';r=' + search_radius
    geocode_url += '&pretty'

    # print(geocode_url)

    while True:
        try:
            here_query = requests.get(geocode_url).json()
            address = here_query['search']['context']['location']['address']['text'].replace('<br/>', ', ')
            # print(address)
            # time.sleep(1)
            return address

        except requests.exceptions.ConnectionError:
            print('Connection Error. Pausing query for {} minutes...'.format(wait_time))
            time.sleep(wait_time * 60)

        except:
            return 'Singapore'



def extract_placetype(data):
    """
    Extracts potential place type information based on several key words: highway,
    amenity, landuse, shop.
    :param property_dict: tag information for the POI obtained from OSM
    :return: list of place type strings
    """
    place_types = []

    if data.fclass is not None:
        place_types.append(data.fclass)

    if 'type' in data.keys() and data.type is not None:
        place_types.append(data.type)

    return place_types


def format_data(data):
    """
    This function takes in the result of the OVERPASS API and formats it
    into a geojson dictionary which will be returned. The dictionary will also be
    saved as a local json file.
    """
    def swap_coord(coordinates_list):
        return [[lnglat_pair[1], lnglat_pair[0]] for lnglat_pair in coordinates_list]

    # print(data)
    if data.geometry is None or data['name'] is None:
        # print('Missing geometry or name information')
        return None

    # print(data.geometry.geom_type)

    # Extract geometry information
    if data.geometry.geom_type is 'Point':
        lng, lat = data.geometry.x, data.geometry.y
        geometry = {'location': {'type': 'Point',
                                 'coordinates': [lat, lng]}}
    elif data.geometry.geom_type is 'Polygon':
        lng, lat = data.geometry.centroid.x, data.geometry.centroid.y

        geometry = {'location': {'type': 'Point',
                                 'coordinates': [lat, lng]},
                    'bound': {'type': data.geometry.geom_type,
                              'coordinates': swap_coord(list(data.geometry.exterior.coords))}}
    elif data.geometry.geom_type is 'MultiPolygon':
        lng, lat = data.geometry.centroid.x, data.geometry.centroid.y

        geometry = {'location': {'type': 'Point',
                                 'coordinates': [lat, lng]},
                    'bound': {'type': data.geometry.geom_type,
                              'coordinates': swap_coord(list(data.geometry[0].exterior.coords))}}
    else:
        raise ValueError('{} is not supported'.format(data.geometry.geom_type))

    # Extract osm address from here map
    address = query_address(lat, lng)

    # Extract place type
    place_type = extract_placetype(data)

    poi_dict = {
        'type': 'Feature',
        'geometry': geometry,
        'properties': {'address': segment_address(address),
                       'name': data['name'],
                       'place_type': place_type,
                       'source': 'OpenStreetMap',
                       'requires_verification': {'summary': 'No'}},
        'id': str(data.osm_id),
        'extraction_date': extract_date()
    }

    # print(poi_dict)
    # print()

    return [poi_dict]


def within_boundary(data, shapefile):
    if data.geometry.geom_type is 'Point':
        lng, lat = data.geometry.x, data.geometry.y
    elif data.geometry.geom_type is 'Polygon' or data.geometry.geom_type is 'MultiPolygon':
        lng, lat = data.geometry.centroid.x, data.geometry.centroid.y
    else:
        raise ValueError('{} is not supported'.format(data.geometry.geom_type))

    within = int(np.sum(shapefile['geometry'].apply(lambda x: Point(lng, lat).within(x))))
    if within == 0:
        return False
    elif within > 0 and within <= 5:
        return True
    else:
        raise ValueError('Data point can be found in more than 5 areas')


if __name__ == '__main__':
    here_app_key = 'ChgzzPNIMr-lHVXDqgEFpuV9HbOwLzcB5SCxHpy_l8s'
    wait_time = 1
    output_filename = 'osm_poi_v2.json'

    # Import shapefile for Singapore
    singapore_shp = gpd.read_file('master-plan-2014-region-boundary-no-sea-shp/MP14_REGION_NO_SEA_PL.shp')
    singapore_shp = singapore_shp.to_crs(epsg="4326")

    # Import shape file for OSM POI data
    filenames = ['gis_osm_buildings_a_free_1.shp', 'gis_osm_pois_a_free_1.shp', 'gis_osm_pois_free_1.shp']
    # filenames = ['gis_osm_pois_free_1.shp']

    for filename in filenames:
        data = gpd.read_file('malaysia-singapore-brunei-latest-free.shp/' + filename)
        data = data.to_crs(epsg="4326")

        # Extract POI data
        for i in tqdm(range(len(data))):
            if filename == 'gis_osm_buildings_a_free_1.shp':
                continue

            if within_boundary(data.iloc[i], singapore_shp):
                formatted_poi = format_data(data.iloc[i])
                if formatted_poi is None:
                    continue

                if os.path.exists(output_filename):
                    with open(output_filename) as json_file:
                        feature_collection = json.load(json_file)
                        feature_collection['features'] += formatted_poi

                    with open(output_filename, 'w') as json_file:
                        json.dump(feature_collection, json_file)

                else:
                    with open(output_filename, 'w') as json_file:
                        feature_collection = {'type': 'FeatureCollection',
                                              'features': formatted_poi}
                        json.dump(feature_collection, json_file)
            else:
                continue

    # Remove duplicated information
    with open(output_filename) as json_file:
        feature_collection = json.load(json_file, strict=False)

    print('Initial number of data points: {}'.format(len(feature_collection['features'])))
    feature_collection['features'] = remove_duplicate(feature_collection['features'])
    print('Final number of data points: {}'.format(len(feature_collection['features'])))

    with open(output_filename, 'w') as json_file:
        json.dump(feature_collection, json_file)
