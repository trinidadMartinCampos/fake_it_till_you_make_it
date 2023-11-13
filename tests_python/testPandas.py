# Project: python test pandas
# Author: Trinidad Martín Campos
# Created: November 6, 2023
# Description: This script performs a simple test for python pandas

def extractData(pandas_obj, columns_list, diameter, condition):
    """Python statement to keep only specific columns.

    Params:
        pandas_obj: Pandas from csv file,
            The input file
        columns_list: List of columns names to filter
        diameter: float
            condition for est_diameter_min
        condition: boolean
            condition for hazardous

    Returns:
        A pandas with filtered columns.
    """
    extracted_pandas = pandas_obj.loc[:, columns_list]
    # extract columns where the minimum estimated diameter is larger than 3.5 kilometers and 'hazardous' is True
    filtered_pandas = extracted_pandas.loc[
        (extracted_pandas['est_diameter_min'] > diameter) & (extracted_pandas['hazardous'] == condition)]
    return filtered_pandas


def extractDataPretty(pandas_obj, categories, gear_box):
    """Python statement to keep only specific columns.

    Params:
        pandas_obj: Pandas from csv file,
            The input file
        categories: List of str
            'Sedan', 'Jeep', 'Coupe'
        gear_box: list of str
            condition manual, automatic

    Returns:
        A pandas with filtered columns.
    """
    condition_1 = pandas_obj['Category'].isin(categories)
    condition_2 = pandas_obj['Leather_interior'] == 'Yes'
    condition_3 = pandas_obj['Gear_box_type'].isin(gear_box)

    data_extracted = pandas_obj.loc[condition_1 & condition_2 & condition_3]
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
    data_input = pd.read_csv(
        'https://codefinity-content-media.s3.eu-west-1.amazonaws.com/4bf24830-59ba-4418-969b-aaf8117d522e/planet')
    names_input = ['name', 'est_diameter_min', 'hazardous']
    print(extractData(data_input, names_input, 3.5, True))

    # Pretty version
    data_input = pd.read_csv(
        'https://codefinity-content-media.s3.eu-west-1.amazonaws.com/4bf24830-59ba-4418-969b-aaf8117d522e/cars',
        index_col=0)
    categories_input = ['Sedan', 'Jeep', 'Coupe']
    gear_box_input = ['Variator', 'Automatic']
    print(extractDataPretty(data_input, categories_input, gear_box_input))

    # Between function
    data_input = pd.read_csv(
        'https://codefinity-content-media.s3.eu-west-1.amazonaws.com/4bf24830-59ba-4418-969b-aaf8117d522e/cars',
        index_col=0)
    fuel_types_input = ['Plug-in Hybrid', 'Hybrid']
    print(betweenFunction(data_input, fuel_types_input).head())

    # Groupby function
    data_input = pd.read_csv(
        'https://codefinity-content-media.s3.eu-west-1.amazonaws.com/4bf24830-59ba-4418-969b-aaf8117d522e/plane',
        index_col=0)
    # Output the first 10 rows of the data set
    print(groupByFunction(data_input).head(10))

    # aggregate
    data_input = pd.read_csv(
        'https://codefinity-content-media.s3.eu-west-1.amazonaws.com/4bf24830-59ba-4418-969b-aaf8117d522e/plane',
        index_col=0)
    data_flights_input = data_input.groupby(['AirportFrom', 'AirportTo']).agg(
        {'Time': ['mean', 'max'], 'Length': 'median'})
    print(data_flights_input.head(10))

    # pivot table approach
    data_input = pd.read_csv(
        'https://codefinity-content-media.s3.eu-west-1.amazonaws.com/4bf24830-59ba-4418-969b-aaf8117d522e/plane',
        index_col=0)
    data_flights_input = pd.pivot_table(data_input, index=['Airline', 'AirportFrom'],
                                        values=['Delay', 'Length'],
                                        aggfunc={'Length': ['min', 'max'], 'Delay': 'count'})

    print(data_flights_input.head(10))
