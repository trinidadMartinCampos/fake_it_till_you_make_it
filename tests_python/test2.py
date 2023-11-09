# Project: python test 2 
# Author: Trinidad Martín Campos
# Created: November 6, 2023
# Description: This script performs a simple test for python string manipulation.


def trimming(ages):
    '''Python statement to remove y/o, convert to integer and calculare mean of the list

    Params:
        ages: List of str,
            The input list

    Returns:
        The mean of the values in the list.
    '''
    # Loading library
    import numpy as np
    # Iterating over list
    for i in range(len(ages)):
        ages[i] = int(ages[i].strip('y/o'))
    return np.array(ages).mean()

def formatting(name, day):
    '''Python statement to format a string with a pattern
    Params:
        name: str,
        day: day of the week
            
    Returns:
        No returns. Print the formatted pattern with the variables
    '''
    # create pattern
    pattern= "Glad to see you, {}, on this wonderful {}!"
    print (pattern.format(name, day))

def formatting_withOrder(day, country, month):
    '''Python statement to format a string with a pattern
    Params:
        country: str,
        day: str,
        month: str
            
    Returns:
        No returns. Print the formatted pattern with the variables
    '''
    # create pattern
    pattern= "Independence Day in {2} is celebrated on the {0} of {1}."
    print (pattern.format(day, month,country))

def formatting_withDict(dict):
    '''Python statement to format a string with a given dictionary to use within the pattern
    Params:
        dict: dictionary {country, capital}
            
    Returns:
        No returns. Print the formatted pattern with the variables
    '''
    # create pattern
    geo_str = "The capital of {d[country]} is {d[capital]}"
    print (geo_str.format(d=dict))

def challenge(dict_person):
    '''1. Add new value to person dictionary with key bmi and value weight/height**2
       2. Create pattern named info "name weight is weight kg, height is height m. BMI is bmi". Everything there written in italics is the value of a person by respective key.
       3. You need to apply .format() method to info string passing person dictionary as an argument and print the result.
    Params:
        dict: dictionary {name, weight, height}
            
    Returns:
        No returns. Print the formatted pattern with the variables
    '''
    #1. Add new value
    new_value=dict_person['weight']/dict_person['height']**2
    dict_person['bmi']=new_value

    #2. Create pattern
    info= "{d[name]} weight is {d[weight]} kg, height is {d[height]} m. BMI is {d[bmi]}"

    #3. Format
    print(info.format(d=dict_person))

def print_symbols(min_value,max_value):
    '''Python statement to format a string to print + and - symbols
    Params:
        min_value and max_value: float
            
    Returns:
        No returns. Print the formatted pattern with the variables
    '''
    # create pattern
    print ("The minimum temperature is {:+} C, the maximum is {:+} C".format(min_value, max_value))

def formatting_extras(population,area):
    '''Python statement to format a string to print decimals, %
    Params:
        population and area: int, float
            
    Returns:
        No returns. Print the formatted pattern with the variables
    '''
    # Format the first string so the population and area will be printed in format: 9,147,420, and insert variables in the correct order.
    print("Area of USA: {0:,} sq.km. Population: {1:,}".format(area, population))
    # Within the second .format function calculate the population density and format the number in format 28.45.
    print("Population density: {0:.2f} people/sq.km".format(population/area))

def formatting_extras2(population,urban_pop):
    '''Python statement to format a string to print %
    Params:
        population and area: int, int
            
    Returns:
        No returns. Print the formatted pattern with the variables
    '''
    # Print total and urban populations with commas every thousand;
    print("USA Population: {0:,}, {1:,} of them - urban population.".format(population, urban_pop))
    # Calculate urban population (%), and print the result in the following format: 45.653%.
    print("Urban population: {0:.3%}".format(urban_pop/population))



if __name__ == "__main__":
    # Code to be executed when the script is run directly
    #trim function
    ages = ['43 y/o', '24 y/o', '34 y/o', '23 y/o']
    print(trimming(ages))

    #formatting
    formatting('Trini', 'Thursday')
    formatting_withOrder('4th', 'USA','July')
    dict_example={'country': 'the United States', 'capital': 'Washington D.C.'}
    formatting_withDict(dict_example)
    dict_person={'name': 'John', 'weight': 76, 'height': 1.79}
    challenge(dict_person)

    #print symbols
    min_temp = -40
    max_temp = 42
    print_symbols(min_temp,max_temp)

    #formatting decimals
    population = 331002651
    area = 9147420
    formatting_extras(population,area)

    #formatting %
    population = 331002651
    urban_pop = 273975139
    formatting_extras2(population,urban_pop)

    