#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import folium
import time
import matplotlib.pyplot as plt
from tqdm import tqdm 
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment


# In[2]:


districts = {
    'Chennai': (13.0836939, 80.270186),
    'Coimbatore': (11.0018115, 76.9628425),
    'Madurai': (9.9261153, 78.1140983),
    'Tiruchirappalli': (10.804973, 78.6870296),
    'Salem': (11.6551982, 78.1581771),
    'Tirunelveli': (8.7284271, 77.7112925),
    'Tiruppur': (11.1017815, 77.345192),
    'Vellore': (12.7948109, 79.0006410968549),
    'Thoothukudi': (8.8052602, 78.1452745),
    'Erode': (11.3306483, 77.7276519),
    'Thanjavur': (10.7860267, 79.1381497),
    'Dindigul': (10.3303299, 78.0673979084697),
    'Cuddalore': (11.7564329, 79.7634644),
    'Kanchipuram': (12.9647163, 79.9839686),
    'Kanyakumari': (8.079252, 77.5499338),
    'Karur': (10.9596041, 78.0807797),
    'Krishnagiri': (12.5188835, 78.2206536),
    'Nagapattinam': (10.805627600000001, 79.824659783024),
    'Namakkal': (11.2191692, 78.16787),
    'Nilgiris': (11.031996, 77.2566384),
    'Perambalur': (11.2287716, 78.8182555496278),
    'Pudukkottai': (10.5, 78.833333),
    'Ramanathapuram': (9.3895523, 78.85907071521498),
    'Sivaganga': (9.851231, 78.53047154820717),
    'Tenkasi': (9.031895800000001, 77.36536124793122),
    'Theni': (9.969664300000002, 77.47420048524822),
    'Thiruvarur': (10.73618605, 79.63318659437627),
    'Tirupathur': (12.453306399999999, 78.55290857022919),
    'Tiruvallur': (13.13014755, 79.92435386254968),
    'Tiruvannamalai': (12.22713775, 79.07012882091152),
    'Villupuram': (11.9398285, 79.4945645),
    'Virudhunagar': (9.58224, 77.9537)
}


# In[3]:


# Function to calculate distance matrix
def calculate_distance_matrix(districts):
    num_districts = len(districts)
    distance_matrix = np.zeros((num_districts, num_districts))
    district_names = list(districts.keys())

    for i in range(num_districts):
        for j in range(num_districts):
            if i != j:
                distance_matrix[i][j] = geodesic(districts[district_names[i]], districts[district_names[j]]).km
    
    return distance_matrix, district_names


# In[4]:


# Implementing the ACO Algorithm
class AntColonyOptimization:
    def __init__(self, distance_matrix, num_ants, num_iterations, alpha=1.0, beta=2.0, rho=0.1):
        self.distance_matrix = distance_matrix
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.num_nodes = distance_matrix.shape[0]
        self.pheromone_matrix = np.ones((self.num_nodes, self.num_nodes))

    def _select_next_node(self, current_node, visited):
        unvisited = list(set(range(self.num_nodes)) - set(visited))
        pheromones = np.array([self.pheromone_matrix[current_node][j] for j in unvisited])
        heuristics = np.array([1.0 / (self.distance_matrix[current_node][j] + 1e-10) for j in unvisited])
        probabilities = (pheromones ** self.alpha) * (heuristics ** self.beta)
        probabilities /= probabilities.sum()
        next_node = np.random.choice(unvisited, p=probabilities)
        return next_node

    def _update_pheromones(self, all_solutions, all_costs):
        self.pheromone_matrix *= (1 - self.rho)
        for solution, cost in zip(all_solutions, all_costs):
            for i in range(len(solution) - 1):
                self.pheromone_matrix[solution[i], solution[i + 1]] += 1.0 / cost

    def run(self):
        best_solution = None
        best_cost = float('inf')
        for _ in range(self.num_iterations):
            all_solutions = []
            all_costs = []
            for _ in range(self.num_ants):
                solution = [0]  # Start from Chennai (assuming Chennai is the first district)
                while len(solution) < self.num_nodes:
                    next_node = self._select_next_node(solution[-1], solution)
                    solution.append(next_node)
                solution.append(0)  # Return to Chennai
                cost = sum(self.distance_matrix[solution[i], solution[i + 1]] for i in range(len(solution) - 1))
                all_solutions.append(solution)
                all_costs.append(cost)
                if cost < best_cost:
                    best_cost = cost
                    best_solution = solution
            self._update_pheromones(all_solutions, all_costs)
        return best_solution, best_cost


# In[9]:


# Parameters (This can be modified based on the computational power and problem complexity)
num_runs = 10
num_ants = 20
num_iterations = 100

# Calculate distance matrix and district names
distance_matrix, district_names = calculate_distance_matrix(districts)

# Perform multiple runs of ACO with tqdm progress bar
best_costs_per_iteration = np.zeros((num_runs, num_iterations))

# Perform multiple runs of ACO with tqdm progress bar
results = []
best_costs = []
for _ in tqdm(range(num_runs)):
    # Initialize ACO instance
    aco = AntColonyOptimization(distance_matrix, num_ants, num_iterations)
    
    # Run ACO algorithm
    best_solution, best_cost = aco.run()
    
    # Store results
    results.append((best_solution, best_cost))
    
     # Store best cost per iteration
    best_costs.append(best_cost)


# In[10]:


print(best_costs)


# In[11]:


# Convert best_costs to a numpy array for easier manipulation (optional)
best_costs_array = np.array(best_costs)

# Calculate the number of iterations
num_iterations = len(best_costs)

# Plotting the best costs over iterations
plt.figure(figsize=(10, 6))
plt.plot(np.arange(num_iterations), best_costs_array, marker='o', linestyle='-', color='b', label='Best Cost')

# Add labels and title
plt.xlabel('Iteration')
plt.ylabel('Best Cost')
plt.title('Best Cost per Iteration')

# Display grid
plt.grid(True)

# Add legend
plt.legend()

# Show plot
plt.tight_layout()
plt.show()


# In[12]:


# Analyze results
best_costs = [cost for _, cost in results]
mean_cost = np.mean(best_costs)
median_cost = np.median(best_costs)
std_deviation = np.std(best_costs)
min_cost = np.min(best_costs)
max_cost = np.max(best_costs)

print(f"Mean Cost: {mean_cost}")
print(f"Median Cost: {median_cost}")
print(f"Standard Deviation: {std_deviation}")
print(f"Minimum Cost: {min_cost}")
print(f"Maximum Cost: {max_cost}")


# In[14]:


# Create a map centered on Tamil Nadu
map_center = (10.8505, 77.5021)  # Latitude and longitude of Tamil Nadu
map_tn = folium.Map(location=map_center, zoom_start=7)

# Add markers for each district with city names as labels
for i, coords in enumerate(best_solution_coords):
    folium.CircleMarker(
        location=coords,
        radius=5,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.7,
        tooltip=district_names[best_solution[i]]
    ).add_to(map_tn)

# Add lines to show the route
folium.PolyLine(
    locations=best_solution_coords,
    color='red',
    weight=2.5,
    opacity=0.9
).add_to(map_tn)

# Fit map to bounds
bounds = map_tn.get_bounds()
map_tn.fit_bounds(bounds)

# Save the map
map_tn.save("tsp_tamilnadu.html")


# In[ ]:




