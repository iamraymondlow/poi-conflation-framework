import json
from gensim.models.fasttext import load_facebook_vectors as load_vectors


def calculate_similarity_score(placetypes):
    """
    This function takes in a list of place types and maps them to their
    corresponding place types based on Google's place type categorisation.
    :param placetype_list: list containing the different placetypes
    :return: list containing the mapped placetypes from google
    """
    scores = []
    new_placetypes = []

    for placetype in placetypes:
        new_placetype = ''
        top_score = 0.0
        for google_placetype in google_placetypes:
            sim_score = round(word_vectors.similarity(placetype, google_placetype), 3)
            if sim_score > top_score:
                top_score = sim_score
                new_placetype = google_placetype

        scores.append(top_score)
        new_placetypes.append(new_placetype)

    print('Original placetype: {}'.format(placetypes))
    print('Similarity score: {}'.format(scores))
    print('Mapped placetype: {}'.format(new_placetypes))
    print()

    index_list = [i for i in range(len(scores)) if scores[i] > cutoff_score]

    if len(index_list) != 0:  # able to find place type mapping that exceeds cutoff score
        return [new_placetypes[index] for index in index_list]
    else:  # unable to find place type mapping that exceeds cutoff score
        duplicates_list = [placetype for placetype in set(new_placetypes) if new_placetypes.count(placetype) > 1]
        return duplicates_list  # return empty list if there are no duplicates


def format_foursquare_placetype(placetype, return_list):
    """
    This function is used to format the place type information from
    Foursquare to match the format used in Google
    :param placetype: place type information in string format
    :param return_list: binary value indicating whether variable:placetype is
    a list or string
    :return: formatted string or list of formatted place type information
    """
    placetype_list = []

    substring_list = placetype.split(' ')

    formatted_placetype = ''
    for substring in substring_list:
        if return_list:
            placetype_list.append(substring.lower())

        formatted_placetype += substring.lower() + '_'

    if len(placetype_list) > 1:  # place type made up of more than one word
        placetype_list.insert(0, formatted_placetype[:-1])
    elif len(placetype_list) == 1:  # place type made up of one word
        pass
    else:  # return a place type string instead of a list
        return formatted_placetype[:-1]

    return placetype_list


def format_heremap_placetype(placetype, return_list):
    """
    This function is used to format the place type information from
    Here Map to match the format used in Google
    :param placetype: place type information in string format
    :param return_list: binary value indicating whether variable:placetype is
    a list or string
    :return: formatted string or list of formatted place type information
    """
    placetype_list = []

    substring_list = placetype.split('-')

    formatted_placetype = ''
    for substring in substring_list:
        if return_list:
            placetype_list.append(substring.lower())

        formatted_placetype += substring.lower() + '_'

    if len(placetype_list) > 1:
        placetype_list.insert(0, formatted_placetype[:-1])
    elif len(placetype_list) == 1:
        pass
    else:
        return formatted_placetype[:-1]

    return placetype_list


def format_sla_placetype(placetype, return_list):
    """
    This function is used to format the place type information from
    the SLA dataset to match the format used in Google
    :param placetype: place type information in string format
    :param return_list: binary value indicating whether variable:placetype is
    a list or string
    :return: formatted string or list of formatted place type information
    """
    placetype_list = []

    if '/' in placetype:
        temp_list = placetype.split('/')
        substring_list = temp_list + placetype.replace('/', ' ').replace(',', '').split(' ')
    else:
        substring_list = placetype.replace('/', ' ').replace(',', '').split(' ')

    formatted_placetype = ''
    for substring in substring_list:
        if return_list:
            placetype_list.append(substring.lower().replace(' ', '_'))

        formatted_placetype += substring.lower() + '_'

    if len(placetype_list) > 1:  # place type made up of more than one word
        placetype_list.insert(0, formatted_placetype[:-1])
    elif len(placetype_list) == 1:  # place type made up of one word
        pass
    else:  # return a place type string instead of a list
        return formatted_placetype[:-1]

    return placetype_list


def format_osm_placetype(placetype, return_list):
    """
    This function is used to format the place type information from
    Open Street Map to match the format used in Google
    :param placetype: place type information in string format
    :param return_list: binary value indicating whether variable:placetype is
    a list or string
    :return: formatted string or list of formatted place type information
    """
    placetype_list = []

    substring_list = placetype.split('_')

    # formatted_placetype = ''
    for substring in substring_list:
        if return_list:
            placetype_list.append(substring.lower())

    if len(placetype_list) > 1:
        placetype_list.insert(0, placetype.lower())
    elif len(placetype_list) == 1:
        pass
    else:
        return placetype.lower()

    return placetype_list


def save_file(map_poi, map_service):
    """
    This function saves the POI information in GeoJSON format
    on the local directory.
    :param poi_list: a GeoJSON dictionary, containing POI information.
    :param map_service: string indicating the source of the POI information.
    :return: None
    """
    with open(map_service + '_poi_mapped.json', 'w') as json_file:
        json.dump(map_poi, json_file)
    print(map_service + '_poi_mapped.json saved.')


def perform_mapping(file_name, map_service):
    """
    This is the main function called to performing the place type mapping
    from some user-defined map service to Google's place type categorisation.
    :param file_name: file name containing the POI information that requires
    place type mapping.
    :param map_service: string indicating the name of the map service.
    :return: list of POI information with the mapped place types.
    """
    with open(file_name) as json_file:
        map_poi = json.load(json_file)

    i = 1
    for feature in map_poi['features']:
        # print('Processing {}: {} / {}'.format(file_name, i, len(map_poi['features'])))
        i += 1
        # print('Original Place Type List: {}'.format(feature['properties']['place_type']))
        # print()
        new_placetypes = []
        for placetype in feature['properties']['place_type']:
            if map_service == 'foursquare':
                new_placetypes_sublist = calculate_similarity_score(
                    format_foursquare_placetype(placetype, return_list=True))
            elif map_service == 'heremap':
                new_placetypes_sublist = calculate_similarity_score(
                    format_heremap_placetype(placetype, return_list=True))
            elif map_service == 'sla':
                new_placetypes_sublist = calculate_similarity_score(
                    format_sla_placetype(placetype, return_list=True))
            elif map_service == 'osm':
                new_placetypes_sublist = calculate_similarity_score(
                    format_osm_placetype(placetype, return_list=True))
            else:
                raise ValueError('Map service not supported.')

            if len(new_placetypes_sublist) != 0:  # place type mapping is successful
                new_placetypes += new_placetypes_sublist
            else:  # place type mapping is unsuccessful
                if map_service == 'foursquare':
                    new_placetypes.append(format_foursquare_placetype(placetype, return_list=False))
                elif map_service == 'heremap':
                    new_placetypes.append(format_heremap_placetype(placetype, return_list=False))
                elif map_service == 'sla':
                    new_placetypes.append(format_sla_placetype(placetype, return_list=False))
                elif map_service == 'osm':
                    new_placetypes.append(format_osm_placetype(placetype, return_list=False))
                else:
                    raise ValueError('Map service not supported.')

                feature['properties']['requires_verification']['summary'] = 'Yes'
                feature['properties']['requires_verification']['reasons'] = ['No appropriate place type mapping: {}'.format(placetype)]

        new_placetypes = list(set(new_placetypes))
        feature['properties']['place_type'] = new_placetypes

        # print('Mapped Place Types: {}'.format(new_placetypes))
        # print()

    return map_poi


if __name__ == '__main__':
    # Load word vectors
    word_vectors = load_vectors('crawl-300d-2M-subword.bin')

    # Define Google placetype
    google_placetypes = ['accounting', 'airport', 'amusement_park', 'aquarium', 'art_gallery', 'atm', 'bakery', 'bank',
                        'bar', 'beauty_salon', 'bicycle_store', 'book_store', 'bowling_alley', 'bus_station', 'cafe',
                        'campground', 'car_dealer', 'car_rental', 'car_repair', 'car_wash', 'casino', 'cemetery',
                        'church', 'city_hall', 'clothing_store', 'convenience_store', 'courthouse', 'dentist',
                        'department_store', 'doctor', 'drugstore', 'electrician', 'electronics_store', 'embassy',
                        'fire_station', 'florist', 'funeral_home', 'furniture_store', 'gas_station', 'gym', 'hair_care',
                        'hardware_store', 'hindu_temple', 'home_goods_store', 'hospital', 'insurance_agency',
                        'jewelry_store', 'laundry', 'lawyer', 'library', 'liquor_store', 'local_government_office',
                        'locksmith', 'lodging', 'meal_delivery', 'meal_takeaway', 'mosque', 'movie_rental',
                        'movie_theater', 'moving_company', 'museum', 'night_club', 'painter', 'park', 'parking',
                        'pet_store', 'pharmacy', 'physiotherapist', 'plumber', 'police', 'post_office',
                        'primary_school', 'real_estate_agency', 'restaurant', 'roofing_contractor', 'rv_park', 'school',
                        'secondary_school', 'shoe_store', 'shopping_mall', 'spa', 'stadium', 'storage', 'store',
                        'subway_station', 'supermarket', 'synagogue', 'taxi_stand', 'tourist_attraction',
                        'train_station', 'transit_station', 'travel_agency', 'university', 'veterinary_care', 'zoo',
                        'administrative_area', 'colloquial_area', 'country', 'establishment', 'finance', 'floor',
                        'food', 'general_contractor', 'geocode', 'health', 'intersection', 'locality',
                        'natural_feature', 'neighborhood', 'place_of_worship', 'political', 'point_of_interest',
                        'post_box', 'postal_code', 'postal_town', 'premise', 'room', 'route', 'street_address',
                        'street_number', 'sublocality', 'subpremise']

    # Define cutoff similarity score
    cutoff_score = 0.95

    # Perform place type mapping
    # foursquare_poi = perform_mapping('foursquare_poi.json', map_service='foursquare')
    # save_file(foursquare_poi, 'foursquare')

    heremap_poi = perform_mapping('heremap_poi.json', map_service='heremap')
    # save_file(heremap_poi, 'heremap')

    sla_poi = perform_mapping('sla_poi.json', map_service='sla')
    # save_file(sla_poi, 'sla')

    osm_poi = perform_mapping('osm_poi.json', map_service='osm')
    # save_file(osm_poi, 'osm')
