# Project: python test 3
# Author: Trinidad Martín Campos
# Created: November 6, 2023
# Description: This script performs a simple test for python data types

def comprehension(list1, given_filter):
    """Python statement to keep only even number of the list

    Params:
        list1: List of int,
            The input list
        given_filter: int

    Returns:
        A list with even numbers.
    """
    filtered_numbers = [num for num in list1 if num % 2 == 0 and num > given_filter]
    return filtered_numbers


def comprehensionDict(dict1, given_filter):
    """Python statement to keep only cities with populations greater than a specified number.

    Params:
        dict1: Dict of str:int,
            The input dict
        given_filter: int

    Returns:
        A dict with filtered cities.
    """
    filtered_cities = {city: population for city, population in cities_population.items() if population > given_filter}
    return filtered_cities


if __name__ == "__main__":
    # Code to be executed when the script is run directly
    # List comprehension
    sample_list = [2, 4, 6, 8, 10, 15, 17, 19, 20]
    threshold = 10
    print(comprehension(sample_list, threshold))

    # Dict comprehension
    cities_population = {
        'New York': 8419600,
        'Los Angeles': 3980400,
        'Chicago': 2716000,
        'Houston': 2328000,
        'Phoenix': 1690000,
        'Smalltown': 15000
    }
    min_population = 5000000
    print(comprehensionDict(cities_population, min_population))
