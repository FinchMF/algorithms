
import os

# ---- FACTORIAL EXAMPLE ----

def factorial(n):

    if n == 0:  
        return 1

    else: 
        return n * factorial(n-1)

# ---- ENGLISH RULER CODE -----

def draw_line(tick_length, tick_label=' '):

    line = '-' * tick_length

    if tick_label:
        line += ' ' + tick_label
    
    print(line)
    return None

def draw_interval(center_length):

    if center_length > 0:

        draw_interval(center_length-1)
        draw_line(center_length)
        draw_interval(center_length-1)

    return None

def draw_ruler(num_inches, major_length):

    draw_line(major_length, '0')
    for j in range(1, 1 + num_inches):

        draw_interval(major_length - 1)
        draw_line(major_length, str(j))

    return None


# ---- BINARY SEARCH ----

def binary_search(data, target, low, high):

    if high >=low:

        mid = (high + low) // 2

        if data[mid] == target:
            return mid

        elif data[mid] > target:

           return binary_search(data, target, low, mid - 1) 

        else:

            return binary_search(data, target, mid + 1, high)

    else: 
        return -1
   
# ---- FILE SYSTEM ----

def disk_usage(path):

    total = os.path.getsize(path)

    if os.path.isdir(path):

        for fname in os.listdir(path):
            childpath = os.path.join(path, fname)
            total += disk_usage(childpath)

    print(f'{total:<7}{path}')
    return total



# ---- FIBONAOCCI ----

def fibonacci_number(n):

    if n <= 1:

        return (n,0)

    else:

        (a,b) = fibonacci_number(n-1)
        return (a+b, a)

def fibonacci(n):

    if n <=1:

        return n
    else:

        return (fibonacci(n - 1) + fibonacci(n - 2))


if __name__ == '__main__':

    print('---- FACTORIAL EXMAPLE -----')
    num = [0, 6, 7, 10, 333]
    for n in num:
        print(f"The factorial of {n} is {factorial(n)}")

    print('\n')
    print('---- ENGLISH RULER EXAMPLE -----')
    num_inches, major_length = [1,2,4], [2,3,4]
    for n in num_inches:
        for ml in major_length:
            print('\n')
            print(f' ------------ Inches: {n} with Major Length: {ml} ')
            print('\n')
            draw_ruler(n, ml)

    print('---- BINARY SEARCH ----')
    data = [2,3,5,6,7,10,55,56,88,100,155,167,200]
    target = 56
    low = 0
    high = len(data) - 1

    res = binary_search(data, target, low, high)

    if res != -1:

        print(f'Element present at idx {res}')

    else:

        print('Element is not present in data')

    print('\n')

    print('---- FILE SYSTEM DISK USAGE ----')

    path = '/Users/finchmf/coding/data_structures_and_algorhitms'
    print(f'Total disk usage: {disk_usage(path)}')
    print('\n')


    print('---- FIBONACCI ----')
    n = 15

    res = fibonacci_number(n)
    print(f'The {n}th and {n-1}th Fibonacci numbers are: {res}')
    print('\n')
    print('Fibonacci Sequence:\n')
    for i in range(n+1):
       print(fibonacci(i))


