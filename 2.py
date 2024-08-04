import pandas as pd

# Read the data from the 'weather.csv' file into a DataFrame
data = pd.read_csv('enjoysport.csv')

# Extract the feature values (concepts) and the target values from the DataFrame
concepts = data.iloc[:, :-1].values  # All rows, all columns except the last one
target = data.iloc[:, -1].values  # All rows, only the last column

# Get the number of features
n = len(concepts[0])

# Initialize the specific hypothesis (S) to the most specific (all '0')
s = ['0'] * n

# Initialize the general hypothesis (G) to the most general (all '?')
g = ['?'] * n

print("Initialization of specific and general hypothesis:")
print("S0", s, "\nG0", g)

def learn(concepts, target):
    # Start with the first example in concepts as the initial specific hypothesis (S)
    s = concepts[0].copy()

    # Initialize the general hypothesis (G) to be as general as possible
    g = [['?' for _ in range(len(s))] for _ in range(len(s))]

    # Iterate over each example and its corresponding target value
    for i, h in enumerate(concepts):
        if target[i] == 'yes':
            # If the target is 'yes', it's a positive example
            print("Positive example", concepts[i])

            # Update the specific hypothesis (S)
            for x in range(len(s)):
                if h[x] != s[x]:
                    s[x] = '?'
                    g[x][x] = '?'
        else:
            # If the target is 'no', it's a negative example
            print("Negative example", concepts[i])

            # Update the general hypothesis (G)
            for x in range(len(s)):
                if h[x] != s[x]:
                    g[x][x] = s[x]
                else:
                    g[x][x] = '?'

        # Print the current state of specific and general hypotheses
        print(s)
        print(g)

    # Remove the redundant hypotheses from the general hypothesis set
    g = [h for h in g if h != ['?' for _ in range(len(s))]]

    # Return the final specific and general hypotheses
    return s, g

# Run the learning algorithm
s_final, g_final = learn(concepts, target)

# Print the final specific hypothesis
print("\nThe Final Specific Hypothesis:")
print(s_final)

print("\nThe Final General Hypothesis:")
# Print the final general hypothesis
print(g_final)
