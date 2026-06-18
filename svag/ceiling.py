import math

nums = [2.8336, 1.1725, 9.4614, 1.4473, 4.7178, 11.541, 54.445, 59.35,
        3.1612, 15.525, 16.814, 2.6103, 7702, 25106, 1596, 1227]

def ceil_3dp(x):
    return math.ceil(x * 1000) / 1000

result = [ceil_3dp(x) for x in nums]

print(result)