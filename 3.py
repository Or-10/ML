import csv
from math import log2

def main():
    file = "prog23.csv"
    data = load_csv(file)
    labels = data[0]
    tree = build_tree(data[1:], labels)
    print(".............Decision Tree...........")
    print(tree)

def load_csv(file):
    with open(file, 'r') as csv_file:
        return [row for row in csv.reader(csv_file)]

def build_tree(data, labels):
    results = [row[-1] for row in data]
    if results.count(results[0]) == len(results):
        return results[0]
    
    best_attr = select_best_attribute(data)
    best_attr_label = labels[best_attr]
    tree = {best_attr_label: {}}
    labels = labels[:best_attr] + labels[best_attr+1:]
    
    for value in set(row[best_attr] for row in data):
        subtree_data = [row[:best_attr] + row[best_attr+1:] for row in data if row[best_attr] == value]
        tree[best_attr_label][value] = build_tree(subtree_data, labels[:]) if subtree_data else results[0]
    
    return tree

def select_best_attribute(data):
    base_entropy = calc_entropy(data)
    best_gain = -1
    best_attr = -1
    
    for attr in range(len(data[0]) - 1):
        subsets = [split_data(data, attr, val) for val in set(row[attr] for row in data)]
        info_gain = base_entropy - sum(len(subset) / len(data) * calc_entropy(subset) for subset in subsets)
        if info_gain > best_gain:
            best_gain, best_attr = info_gain, attr
    
    return best_attr

def split_data(data, attr, value):
    return [row[:attr] + row[attr+1:] for row in data if row[attr] == value]

def calc_entropy(data):
    total = len(data)
    counts = {row[-1]: 0 for row in data}
    for row in data:
        counts[row[-1]] += 1
    
    return -sum((count / total) * log2(count / total) for count in counts.values())

if __name__ == "__main__":
    main()
