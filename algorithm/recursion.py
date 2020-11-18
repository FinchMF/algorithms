
def factorial(n):

    if n == 0:  
        return 1

    else: 
        return n * factorial(n-1)


if __name__ == '__main__':

    num = [0, 6, 7, 10, 333]
    for n in num:
        print(f"The factorial of {n} is {factorial(n)}")