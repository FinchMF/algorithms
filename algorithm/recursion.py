
# ---- FACTORIAL EXAMPLE ----

def factorial(n):

    if n == 0:  
        return 1

    else: 
        return n * factorial(n-1)

# ---- ENGLISH RULLER CODE -----

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
            draw_ruler(n, ml)
            print('\n')
            print(' ------------ ')
            print('\n')

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


