#USING BRANCH AND BOUND

class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value

def knapsack_branch_and_bound(items, capacity):
    def bound(i, weight, value):
        if weight > capacity:
            return 0
        bound_value = value
        j = i
        total_weight = weight
        while j < n and total_weight + items[j].weight <= capacity:
            total_weight += items[j].weight
            bound_value += items[j].value
            j += 1
        if j < n:
            bound_value += (capacity - total_weight) * (items[j].value / items[j].weight)
        return bound_value

    def knapsack_recursive(i, weight, value):
        nonlocal max_value
        if weight <= capacity and value > max_value:
            max_value = value
        if i < n:
            if weight + items[i].weight <= capacity:
                knapsack_recursive(i + 1, weight + items[i].weight, value + items[i].value)
            if bound(i + 1, weight, value) > max_value:
                knapsack_recursive(i + 1, weight, value)

    n = len(items)
    max_value = 0
    items.sort(key=lambda x: x.value / x.weight, reverse=True)
    knapsack_recursive(0, 0, 0)
    return max_value

# Example usage
if __name__ == "__main__":
    items = [Item(10, 60), Item(20, 100), Item(30, 120)]
    capacity = 50

    max_value = knapsack_branch_and_bound(items, capacity)
    print("Maximum value in the knapsack:", max_value)


#USING DYNAMIC PROGRAMMING
def knapsack_dynamic_programming(values, weights, capacity):
    n = len(values)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        for w in range(capacity + 1):
            if i == 0 or w == 0:
                dp[i][w] = 0
            elif weights[i - 1] <= w:
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]

    selected_items = []
    i, w = n, capacity
    while i > 0 and w > 0:
        if dp[i][w] != dp[i - 1][w]:
            selected_items.append(i - 1)
            w -= weights[i - 1]
        i -= 1

    selected_items.reverse()
    return dp[n][capacity], selected_items

# Example usage
if __name__ == "__main__":
    values = [60, 100, 120]
    weights = [10, 20, 30]
    capacity = 50

    max_value, selected_items = knapsack_dynamic_programming(values, weights, capacity)
    print("Maximum value in the knapsack:", max_value)
    print("Selected items:", selected_items)