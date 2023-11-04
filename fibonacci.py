def fibonacci_non_recursive(n):
    if n <= 0:
        return [0]
    
    fibonacci_series = [0, 1]

    while True:
        next_fib = fibonacci_series[-1] + fibonacci_series[-2]
        if len(fibonacci_series) <= n:
            fibonacci_series.append(next_fib)
        else:
            break

    return fibonacci_series

def fibonacci_recursive(n):
    if n <= 0:
        return [0]
    elif n == 1:
        return [0, 1]

    series = fibonacci_recursive(n - 1)
    series.append(series[-1] + series[-2])
    return series

# Get input from the user
n = int(input("Enter the value of n: "))

# Calculate and print the Fibonacci series up to n using the non-recursive method
series_nr = fibonacci_non_recursive(n)
print("Fibonacci series (non-recursive):")
print(series_nr)

# Calculate and print the Fibonacci series up to n using the recursive method
series_r = fibonacci_recursive(n)
print("Fibonacci series (recursive):")
print(series_r)
