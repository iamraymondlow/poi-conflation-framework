import numpy as np
from postal.parser import parse_address
import hashlib
from datetime import datetime


def translate_coordinate(lat, lng, l, h):
    """
    This function takes in a latitude,longitude pair and translates it to produce a l x h area (m^2) with the
    original latitude, longitude pair located at the centroid.
    """
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


def identify_centroid(max_lat, max_lng, min_lat, min_lng):
    """
    This function takes in the coordinates of the bounding box and outputs its centroids.
    """
    return (max_lat + min_lat) / 2.0, (max_lng + min_lng) / 2.0


def calculate_circle_radius(max_lat, max_lng, centre_lat, centre_lng):
    """
    This function takes in the edges of a bounding box and its centroid to calculate the radius of a bounding
    circle that can be fitted within the bounding box with the centroid at its centre.
    """
    max_coordinate_dist_lat = abs(max_lat - centre_lat)
    max_coordinate_dist_lng = abs(max_lng - centre_lng)

    # Define the Earth’s radius, assuming a spherical surface
    earth_radius = 6378137.0

    lng_radius = max_coordinate_dist_lng * (earth_radius * np.cos(np.pi * centre_lat / 180.0)) / (180.0 / np.pi)
    lat_radius = max_coordinate_dist_lat * earth_radius / (180.0 / np.pi)

    if lng_radius >= lat_radius:
        print('Radius of bounding circle (Latitude): {}'.format(lat_radius))
        return lat_radius
    else:
        print('Radius of bounding circle (Longitude): {}'.format(lng_radius))
        return lng_radius


def calculate_lat_lng_distance(min_coordinate, max_coordinate, is_lat=True, latitude=None):
    """
    This function calculates the distance (in metres) between two coordinates. Calculations differ depending if the
    coordinate pair comes from the latitude or longitude.
    """
    # Define the Earth’s radius, assuming a spherical surface
    radius = 6378137.0

    if is_lat:
        return (max_coordinate - min_coordinate) * radius / (180.0 / np.pi)
    else:
        return (max_coordinate - min_coordinate) * (radius * np.cos(np.pi * latitude / 180.0)) / (180.0 / np.pi)


def generate_coordinate_list(min_coordinate, max_coordinate, num_box):
    """
    This function takes in a pair of coordinates (either from the latitude or longitude) and breaks the entire range
    into multiple intervals, whereby each interval is the size of one side of the query box. The coordinates marking
    each interval is then stored in a list and returned as an output.
    """
    query_range = max_coordinate - min_coordinate
    query_interval_length = query_range / num_box

    coordinate_list = []
    coordinate = min_coordinate
    for i in range(int(num_box) + 1):
        coordinate_list.append(coordinate)
        coordinate += query_interval_length

    return coordinate_list


def generate_bounding_coordinates(lat_list, lng_list):
    """
    This function takes in two lists containing coordinates in the latitude and longitude direction and generates a
    list containing coordinates for each bounding box.
    """
    lat_list = list(reversed(lat_list))
    coordinate_list = []
    for i in range(len(lng_list) - 1):
        for j in range(len(lat_list) - 1):
            coordinate_list.append([lat_list[j + 1], lng_list[i], lat_list[j], lng_list[i + 1]])

    return coordinate_list


def divide_bounding_box(max_lat, min_lat, max_lng, min_lng, querybox_dim):
    """
    This function divides the bounding box (defined using its coordinates) up into smaller sub-bounding boxes and
    returns a list of query box coordinates.
    """
    lat_dist = calculate_lat_lng_distance(min_coordinate=min_lat, max_coordinate=max_lat)
    lng_dist = calculate_lat_lng_distance(min_coordinate=min_lng, max_coordinate=max_lng, is_lat=False,
                                          latitude=(max_lat + min_lat) / 2.0)
    num_box_lat = round(lat_dist / querybox_dim)
    num_box_lng = round(lng_dist / querybox_dim)
    lat_list = generate_coordinate_list(min_coordinate=min_lat, max_coordinate=max_lat, num_box=num_box_lat)
    lng_list = generate_coordinate_list(min_coordinate=min_lng, max_coordinate=max_lng, num_box=num_box_lng)
    lat_lng_list = generate_bounding_coordinates(lat_list=lat_list, lng_list=lng_list)

    return lat_lng_list


def generate_id(poi_dict):
    """
    Generates an unique ID based on the encoded string of the POI's name, address and location information.
    """
    poi_string = poi_dict['properties']['name'] + \
                 poi_dict['properties']['address']['formatted_address'] + \
                 str(poi_dict['geometry']['location']['coordinates'][0]) + \
                 str(poi_dict['geometry']['location']['coordinates'][1])
    hasher = hashlib.md5()
    hasher.update(poi_string.encode('utf-8'))
    return hasher.hexdigest()


def within_boundary_area(lat, lng, min_lat, max_lat, min_lng, max_lng):
    """
    Checks if the latitude and longitude pair falls within a region of interest.
    """
    if lat > max_lat or lat < min_lat:
        return False
    if lng > max_lng or lng < min_lng:
        return False

    return True


def extract_date():
    """
    Extract the date when the query is made in the format YYYYMMDD.
    """
    return datetime.today().strftime('%Y%m%d')


def segment_address(address):
    """
    Parse the address string using the libpostal library and segment it into the various components.
    """
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


def remove_duplicate(poi_data):
    """
    Remove duplicated POIs in the dataset using the POI's unique ID.
    """
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
