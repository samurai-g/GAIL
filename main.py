import pandas as pd
import random
import math
import functools

# Load the CSV file containing the European cities
cities_df = pd.read_csv('european-cities.csv')

# Display the first few rows to understand the structure
print(cities_df.head())

def to_radians(degrees):
    return degrees * math.pi / 180.0

@functools.lru_cache(maxsize=None)
def calculate_distance(lat1, lon1, lat2, lon2):
    EarthRadius = 6371000.0  # Radius of the Earth in meters
    
    phi1 = to_radians(lat1)
    phi2 = to_radians(lat2)
    delta_phi = to_radians(lat2 - lat1)
    delta_lambda = to_radians(lon2 - lon1)
    
    a = (math.sin(delta_phi / 2) * math.sin(delta_phi / 2) +
         math.cos(phi1) * math.cos(phi2) *
         math.sin(delta_lambda / 2) * math.sin(delta_lambda / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = EarthRadius * c
    
    return distance

def initialize_population(size=50):
    population = []
    for _ in range(size):
        route = list(range(len(cities_df)))  # Use indices of the cities DataFrame
        random.shuffle(route)
        population.append(route)
    return population

def route_distance(route):

    distance_meters = 0
    distance_km = 0
    total_distance_km = 0

    #Calculate the distance for each route city[0] to city[1], city[1] to city[2],...
    for i in range(100):
        j = (i + 1) % 100 # Wrap around to the start for the last city
        city_a_index = route[i]
        city_b_index = route[j]
        city_a = cities_df.iloc[city_a_index]
        city_b = cities_df.iloc[city_b_index]
        #Calculate distance with Haversine
        distance_meters = calculate_distance(city_a['latitude'], city_a['longitude'],
                                             city_b['latitude'], city_b['longitude'])
        distance_km = distance_meters / 1000  # Convert distance to kilometers
        #DEBUG: Distances for each trip
        #print("City " + str(i) + " -> " + str(j) + ": " + str(distance_km))
        total_distance_km += distance_km

    return total_distance_km


# DEBUG: usage until here
#population = initialize_population(5)  # Smaller population is faster
#total_distances = [route_distance(route) for route in population]

def tournament_selection(population, distances, tournament_size=5):
    tournament_indices = random.sample(range(len(population)), tournament_size)
    best_index = min(tournament_indices, key=lambda i: distances[i])
    return population[best_index]

def ordered_crossover(parent1, parent2):
    #"""Performs ordered crossover between two parents."""
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)
    
    # Copy a part of parent1's route to the child
    child[start:end+1] = parent1[start:end+1]
    
    # Fill the remaining positions with cities from parent2
    parent2_cities = [city for city in parent2 if city not in child]
    child = [city if city is not None else parent2_cities.pop(0) for city in child]
    
    return child

def mutate(route, mutation_rate=0.01):
    #"""Performs a swap mutation on a route."""
    for i in range(len(route)):
        if random.random() < mutation_rate:
            swap_index = random.randint(0, len(route) - 1)
            route[i], route[swap_index] = route[swap_index], route[i]
    return route

def calculate_generation_distances(population):
    return [route_distance(route) for route in population]

def next_generation(current_gen, distances, elite_size=5, mutation_rate=0.01):
    new_generation = []
    population_size = len(current_gen)
    
    # Elite selection
    elite_indices = sorted(range(population_size), key=lambda i: distances[i])[:elite_size]
    for i in elite_indices:
        new_generation.append(current_gen[i])
    
    # Generating new offspring
    while len(new_generation) < population_size:
        parent1 = tournament_selection(current_gen, distances)
        parent2 = tournament_selection(current_gen, distances)
        child = mutate(ordered_crossover(parent1, parent2), mutation_rate)
        new_generation.append(child)

    # Calculate distances for the new generation
    new_distances = calculate_generation_distances(new_generation)
    return new_generation, new_distances

# DEBUG: usage until here
# Initialize first generation
population = initialize_population(50)
distances = calculate_generation_distances(population)

# Run the genetic algorithm for a number of generations
num_generations = 100
for _ in range(num_generations):
    population, distances = next_generation(population, distances, elite_size=15, mutation_rate=0.01)

    # Find the best route in the final generation
    #best_route_index = min(range(len(population)), key=lambda i: route_distance(population[i]))
    #best_route = population[best_route_index]
    #best_distance = route_distance(best_route)

    # Find the best route in the current generation using the cached distances
    best_route_index = min(range(len(population)), key=lambda i: distances[i])
    best_distance = distances[best_route_index]

    #print("Best route:" + str(best_route))
    print("Best distance on run nr." + str(_+1)+ ": " + str(best_distance))



