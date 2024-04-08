import pandas as pd
import numpy as np
import random
import math
import functools
import time
import argparse

start_time = time.time()

# Load the CSV file containing the European cities
cities_df = pd.read_csv('european-cities.csv')
#cities_df = pd.read_csv(args.cities)

# Display the first few rows to debug
#print(cities_df.head())

def parse_args():
    parser = argparse.ArgumentParser(description='Solve the Traveling Salesperson Problem with a Genetic Algorithm.')
    # Set default CSV file location to 'cities.csv'. Adjust the default path as necessary.
    parser.add_argument('--cities', type=str, default='cities.csv', help='CSV file containing city coordinates. Default is "cities.csv".')
    # Set default output file location to 'route.txt'. Adjust the default path as necessary.
    parser.add_argument('--output', type=str, default='route.txt', help='Text file to save the calculated route. Default is "route.txt".')
    # If not specified, do not limit the search (use a very high value to simulate "no limit").
    parser.add_argument('--limit', type=float, default=float(2000), help='Terminate when the shortest route is <= this length in kilometers. Default is no limit.')
    # The absence of --secondary implies it's not used, so no default change needed.
    parser.add_argument('--secondary', action='store_true', help='Run with the secondary optimality criterion. Default is False.')
    return parser.parse_args()


def to_radians(degrees):
    return degrees * math.pi / 180.0

def initialize_population(size=10):
    population = []
    for _ in range(size):
        route = list(range(len(cities_df)))  # Use indices of the cities DataFrame
        random.shuffle(route)
        population.append(route)
    return population

#Calculate distance with haversine
@functools.lru_cache(maxsize=None) #this does nothing(?)
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

def calculate_distance_matrix(cities_df):
    """Calculates and returns a matrix of distances between cities."""
    num_cities = len(cities_df)
    distance_matrix = np.zeros((num_cities, num_cities))

    for i in range(num_cities):
        for j in range(i + 1, num_cities):  # Avoid redundant calculations
            # Correctly access latitude and longitude for cities i and j
            lat1 = cities_df.iloc[i]['latitude']
            lon1 = cities_df.iloc[i]['longitude']
            lat2 = cities_df.iloc[j]['latitude']
            lon2 = cities_df.iloc[j]['longitude']

            # Calculate distance using the corrected function signature
            dist = calculate_distance(lat1, lon1, lat2, lon2)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Symmetric matrix

    return distance_matrix


def evaluate_route(route, distance_matrix):
    total_distance = 0
    max_consecutive_distance = 0
    for i in range(len(route)):
        j = (i + 1) % len(route)  # Ensure we loop back to the start for a round trip

        # Access the precomputed distance directly from the distance_matrix
        distance = distance_matrix[route[i], route[j]]
        
        total_distance += distance / 1000  # convert to km
        max_consecutive_distance = max(max_consecutive_distance, distance) / 1000 # longest single leg, also km
    return total_distance, max_consecutive_distance


def fitness_function(route, distance_matrix, optimize_for_max_distance=False):
    total_distance, max_consecutive_distance = evaluate_route(route, distance_matrix)
    if optimize_for_max_distance:
        # Apply a strategy to penalize routes with large max distances between cities
        # This is a simplistic approach; adjust based on experimentation
        penalty = max_consecutive_distance * 0.5  # Example penalty factor
        return total_distance + penalty
    else:
        return total_distance

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

def next_generation(current_gen, distances, elite_size, mutation_rate, distance_matrix, optimize_max_distance):
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
    new_distances = [fitness_function(route, distance_matrix, optimize_max_distance) for route in new_generation]
    return new_generation, new_distances



def main():

    args = parse_args()

    # DEBUG: usage until here
    optimize_max_distance = args.secondary  # depending on user input TO-DO: command line args

    # Calculate distance matrix
    distance_matrix = calculate_distance_matrix(cities_df)

    # Initialize first generation
    population_size = 100
    population = initialize_population(population_size)
    distances = [fitness_function(route, distance_matrix, optimize_max_distance) for route in population]

    # Run the genetic algorithm for a number of generations
    num_generations = 500
    elite_size=15
    mutation_rate=0.01

    print("Population size: " + str(population_size))
    print("Num of generations: " + str(num_generations))
    print("Elite size: " + str(elite_size))
    print("Mutation rate: " + str(mutation_rate))

    for generation in range(num_generations):
        population, distances = next_generation(population, distances, elite_size, mutation_rate, distance_matrix, optimize_max_distance)

        # Find the best route in the final generation
        #best_route_index = min(range(len(population)), key=lambda i: route_distance(population[i]))
        #best_route = population[best_route_index]
        #best_distance = route_distance(best_route)

        # Find the best route in the current generation using the cached distances
        best_route_index = min(range(len(population)), key=lambda i: distances[i])
        best_route = population[best_route_index]
        best_distance, longest_single_leg = evaluate_route(best_route, distance_matrix)

        #print("Best route:" + str(best_route))
        if generation % 50 == 0:
            print(f"Generation {generation}: Shortest roundtrip in the best route: {best_distance}")
            print(f"Generation {generation}: Longest leg in the best route: {longest_single_leg}")

        if args.limit and best_distance <= args.limit:
            print(f"Terminating early: Found a route shorter than {args.limit} km.")
            break


    end_time = time.time()

    total_time = end_time - start_time

    print("Total execution time: " + str(total_time) + " seconds")

    # Save the calculated route to the output file
    with open(args.output, 'w') as f:
        # Here, you would format and write the best_route and best_distance to the file
        f.write(f'Best route: {best_route}\n')
        f.write(f'Total distance: {best_distance} km')

if __name__ == '__main__':
    main()



