def RandomValue(n,x0):
    '''
    linear congruential random number generator
    n: number of random numbers
    x0: seed
    return: list of length n filled with random numbers from 0 to 1
    '''
    a = 24693
    c = 3517
    k = 2**17
    x_prev = x0
    random_numbers = []
    for i in range(n):
        x = (a * x_prev + c) % k
        x_prev = x
        random_numbers.append(x / k)
    return random_numbers

def randomIndex(n, x0, minIndex):
    '''
    Generates a list of random indices using a linear congruential generator (LCG).

    Parameters:
    - n (int): The total number of data points
    - x0 (int): The seed for the random number generator.
    - minIndex (int): The minimum valid index

    Returns:
    - List[int]: A list of random indices
    '''
    random_numbers = RandomValue(n,x0)
    random_Indexes = []
    for eachNumber in random_numbers:
        random_Indexes.append((int)((n * eachNumber) + minIndex))
    return random_Indexes


