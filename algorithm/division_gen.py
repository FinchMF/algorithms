
def sqrt(n):

    return n**(1/2)

def divisors(num):

    divs = []
    for i in range(1, int(sqrt(num) + 1)):

        if num % i == 0:

            yield i
            if i*i != num:

                divs.append(int(num / i))


    for div in reversed(divs):

        yield div


if __name__ == '__main__':

    print(list(divisors(200)))