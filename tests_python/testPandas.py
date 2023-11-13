# Project: python test 3
# Author: Trinidad Martín Campos
# Created: November 6, 2023
# Description: This script performs a simple test for python pandas

def extractData(pandasObj, columnsList, diameter, condition):
    """Python statement to keep only specific columns.

    Params:
        pandasObj: Pandas from csv file,
            The input file
        columnslist: List of columns names to filter
        diameter: float
            condition for est_diameter_min
        condition: boolean
            condition for hazardous

    Returns:
        A pandas with filtered columns.
    """
    extracted_pandas = pandasObj.loc[:, columnsList]
    # extract columns where the minimum estimated diameter is larger than 3.5 kilometers and 'hazardous' is True
    filtered_pandas = extracted_pandas.loc[
        (extracted_pandas['est_diameter_min'] > diameter) & (extracted_pandas['hazardous'])]
    return filtered_pandas


def extractDataPretty(pandasObj, categories, gear_box):
    """Python statement to keep only specific columns.

    Params:
        pandasObj: Pandas from csv file,
            The input file
        categories: List of str
            'Sedan', 'Jeep', 'Coupe'
        gear_box: list of str
            condition manual, automatic

    Returns:
        A pandas with filtered columns.
    """
    condition_1 = pandasObj['Category'].isin(categories)
    condition_2 = pandasObj['Leather_interior'] == 'Yes'
    condition_3 = pandasObj['Gear_box_type'].isin(gear_box)

    data_extracted = pandasObj.loc[condition_1 & condition_2 & condition_3]
    return data_extracted


def betweenFunction(data, fuel_types):
    """Python statement to keep only specific columns.

    Params:
        pandasObj: Pandas from csv file,
            The input file
        fuel_types: list of str
            condition 'Plug-in Hybrid', 'Hybrid'

    Returns:
        A pandas with filtered columns.
    """
    # Put the condition on the column 'Price'
    condition_1 = data['Price'].between(15000, 20000, inclusive='left')
    # Put the condition on the column 'Year'
    condition_2 = data['Year'].between(2015, 2020, inclusive='neither')
    # Put the condition on the column 'Fuel_type'
    condition_3 = data['Fuel_type'].isin(fuel_types)

    # Unite three conditions
    data_extracted = data.loc[condition_1 & condition_2 & condition_3]
    return data_extracted


def groupByFunction(data) -> dict:
    """Python statement to keep only specific columns.

    Params:
        pandasObj: Pandas from csv file,
            The input file
    Returns:
        A pandas with filtered columns.
    """
    # Extract the columns 'AirportFrom', 'Airline', 'Time', and 'Length' from data
    # Apply the .groupby() function to the previous columns.
    # Within the .groupby() function, put the columns 'AirportFrom' and 'Airline'; the order is crucial
    # Calculate the mean value of the column 'Time'.
    data_flights = data[['AirportFrom', 'Airline', 'Time', 'Length']].groupby(['AirportFrom', 'Airline']).mean()

    # Calculate the sum of two columns: 'Length' and 'Time'. Then find their minimum.
    data_flights = data[['AirportFrom', 'Airline', 'Time', 'Length']].groupby(['AirportFrom', 'Airline']).apply(
        lambda x: (x['Length'] + x['Time']).min())

    return data_flights


if __name__ == "__main__":
    import pandas as pd

    # Code to be extract all rows by only specific columns
    data = pd.read_csv(
        'https://codefinity-content-media.s3.eu-west-1.amazonaws.com/4bf24830-59ba-4418-969b-aaf8117d522e/planet')
    names = ['name', 'est_diameter_min', 'hazardous']
    print(extractData(data, names, 3.5, True))

    # Pretty version
    data = pd.read_csv(
        'https://codefinity-content-media.s3.eu-west-1.amazonaws.com/4bf24830-59ba-4418-969b-aaf8117d522e/cars',
        index_col=0)
    categories = ['Sedan', 'Jeep', 'Coupe']
    gear_box = ['Variator', 'Automatic']
    print(extractDataPretty(data, categories, gear_box))

    # Between function
    data = pd.read_csv(
        'https://codefinity-content-media.s3.eu-west-1.amazonaws.com/4bf24830-59ba-4418-969b-aaf8117d522e/cars',
        index_col=0)
    fuel_types = ['Plug-in Hybrid', 'Hybrid']
    print(betweenFunction(data, fuel_types).head())

    # Groupby function
    data = pd.read_csv(
        'https://codefinity-content-media.s3.eu-west-1.amazonaws.com/4bf24830-59ba-4418-969b-aaf8117d522e/plane',
        index_col=0)
    # Output the first 10 rows of the data set
    print(groupByFunction(data).head(10))

    # aggregate
    data = pd.read_csv(
        'https://codefinity-content-media.s3.eu-west-1.amazonaws.com/4bf24830-59ba-4418-969b-aaf8117d522e/plane',
        index_col=0)
    data_flights = data.groupby(['AirportFrom', 'AirportTo']).agg({'Time': ['mean', 'max'], 'Length': 'median'})
    print(data_flights.head(10))
