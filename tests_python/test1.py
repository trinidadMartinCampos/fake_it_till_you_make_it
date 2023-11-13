# Project: python test 1
# Author: Trinidad Martín Campos
# Created: November 6, 2023
# Description: This script performs a simple test for python basics.

def sum_list(example_list):
    """Python statement to display the sum of the values in the list

    Params:
        example_list: List of float,
            The input list

    Returns:
        The sum of the values in the list.
    """
    return sum(example_list)


def pv(future_value, rate, n):
    """
    Discount a value at defined rate n time periods into the future.

    Formula:
    PV = FV / (1 + r)^n
    Where
    FV = future value
    r = the interest rate
    n = number of years in the future

    Params:
    -------
    future value: float,
        The value to discount

    rate: float
        the rate at which to do the discounting

    n: float,
        the number of time periods into the future

    Returns:
    --------
    float
    """
    return future_value / (1 + rate) ** n


def fizzbuzz(n):
    """Function checking if an integer is multiple of 3, 5 or both

    Params:
        n: Integer,
            The number to check

    Returns result_out:
        "FIZZ" if number is a multiple of 3
        “BUZZ” if number is a multiple of 5
        “FIZZBUZZ” if number is a multiple of 3 AND 5
        Number as a string if else
    """
    result_out = ""
    if n % 3 == 0:
        if n % 5 == 0:
            result_out = "FIZZBUZZ"
        else:
            result_out = "FIZZ"
    elif n % 5 == 0:
        result_out = "BUZZ"
    else:
        result_out = str(n)
    return result_out


def convert_celsius_to_fahrenheit(deg_celsius):
    """
    Convert degrees celsius to fahrenheit
    Returns float value - temp in fahrenheit
    
    Parameters:
    -----------
    deg_celcius: float
        temp in degrees celsius
        
    Returns:
    -------
    float
    """
    return (9 / 5) * deg_celsius + 32


def maximum(number_list):
    """
    Find the maximum value of the list
    
    Parameters:
    -----------
    number_list: list of integer
        numbers to compare
        
    Returns:
    -------
    result_out: max integer
    """
    result_out: int = 0
    for num in number_list:
        if num > result_out:
            result_out = num
    return result_out


def flatten_lists(list_of_lists):
    import itertools
    """
    Convert list of lists into a single list
    
    Parameters:
    -----------
    list_of_lists: list of lists 
        lists to flatten
        
    Returns:
    -------
    list: single list
    """
    return list(itertools.chain(*list_of_lists))


if __name__ == "__main__":
    # Code to be executed when the script is run directly
    # Sum function
    x = [3, 5, 9, 1]
    function_sum = sum_list(x)
    print(function_sum)

    # PV function
    # Test case 1
    future_value_input = 2000
    rate_input = 0.035
    n_input = 5
    result = pv(future_value_input, rate_input, n_input)
    print(f'{result:.2f}')
    # Test case 2
    future_value_input = 350
    rate_input = 0.01
    n_input = 10
    result = pv(future_value_input, rate_input, n_input)
    print(f'{result:.2f}')

    # Fizzbuzz function
    list_numbers = [1, 3, 5, 15, 23]
    for number in list_numbers:
        result = fizzbuzz(number)
        print(result)

    # Celsius function
    celsius = [39.2, 36.5, 37.3, 41.0]
    degrees_f = []
    for value in celsius:
        degrees_f.append(convert_celsius_to_fahrenheit(value))
    print(degrees_f)

    # Max function
    to_search = [0, 1000, 2, 999, 5, 100, 54]
    max_number = maximum(to_search)
    print(max_number)

    # Flatten function
    list_of_lists_input = [[8, 2, 1], [9, 1, 2], [4, 5, 100]]
    final_list = flatten_lists(list_of_lists_input)
    print(final_list)

    # Comics function
    comics = ['Iron-man', 'Captain America', 'Spider-man', 'Thor', 'Deadpool']
    # slice and then print the first and second list items
    print(comics[:2])
    # slices and the print the second to fourth list items
    print(comics[1:4])
    # slice and then print the fourth and fifth list items
    print(comics[-2:])
    # append “Doctor Strange” to the list. Print the updated list
    comics.append("Doctor Strange")
    print(comics)
    # insert “Headpool” before “Deadpool” in the list. Print the updated list
    comics.insert(4, "Headpool")
    print(comics)
    # delete “Iron-man”. Print the updated list
    comics.remove("Iron-man")
    print(comics)
