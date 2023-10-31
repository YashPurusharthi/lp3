# Function to solve the Fractional Knapsack problem
def fractional_knapsack(items, capacity):
    # Calculate the value-to-weight ratio for each item
    for item in items:
        item['ratio'] = item['value'] / item['weight']

    # Sort the items by the value-to-weight ratio in descending order
    items.sort(key=lambda x: x['ratio'], reverse=True)

    total_value = 0.0
    knapsack = []

    for item in items:
        if capacity >= item['weight']:
            # If the entire item can be added, add it and update capacity
            knapsack.append(item)
            total_value += item['value']
            capacity -= item['weight']
        else:
            # If only a fraction of the item can be added, add a fraction
            fraction = capacity / item['weight']
            knapsack.append({'name': item['name'], 'weight': item['weight'] * fraction, 'value': item['value'] * fraction})
            total_value += item['value'] * fraction
            break

    return knapsack, total_value

# Example usage
if __name__ == "__main__":
    items = [
        {'name': 'item1', 'weight': 10, 'value': 60},
        {'name': 'item2', 'weight': 20, 'value': 120},
        {'name': 'item3', 'weight': 30, 'value': 130}
    ]

    max_capacity = 50

    knapsack_items, max_value = fractional_knapsack(items, max_capacity)

    print(f"Maximum value in the knapsack: {max_value}")