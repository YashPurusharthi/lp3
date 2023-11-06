def fractional_knapsack_value(weights, values, capacity):
    n = len(weights)
    value_per_weight = [(values[i] / weights[i], weights[i], values[i]) for i in range(n)]
    value_per_weight.sort(reverse=True, key=lambda x: x[0])
    
    total_value = 0
    knapsack_weight = 0
    
    for i in range(n):
        if knapsack_weight + value_per_weight[i][1] <= capacity:
            knapsack_weight += value_per_weight[i][1]
            total_value += value_per_weight[i][2]
        else:
            fraction = (capacity - knapsack_weight) / value_per_weight[i][1]
            total_value += fraction * value_per_weight[i][2]
            break

    return total_value

# Example usage:
weights = [10, 20, 30]
values = [60, 100, 120]
capacity = 50

result = fractional_knapsack_value(weights, values, capacity)
print(result)
