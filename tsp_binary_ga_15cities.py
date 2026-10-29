
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import tsplib95

# Common parameters
DATASET_PATH = Path(__file__).with_name("a280.tsp")
NUM_CITIES = 15
POPULATION_SIZE = 100
GENERATIONS = 500
TOURNAMENT_SIZE = 5
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.2
ELITISM_COUNT = 2
RANDOM_SEED = 42
BITS_PER_CITY = 4


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "Download a280.tsp and place it beside this Python file."
        )

    problem = tsplib95.load(str(DATASET_PATH))
    cities = list(problem.get_nodes())[:NUM_CITIES]

    distances = {
        (i, j): problem.get_weight(i, j)
        for i in cities
        for j in cities
    }

    coordinates = problem.node_coords
    return cities, distances, coordinates


# Convert a city route into binary chromosomes
def encode_route(route, cities):
    index = {city: i for i, city in enumerate(cities)}
    bits = []

    for city in route:
        bits.extend(
            int(bit)
            for bit in format(index[city], "04b")
        )

    return bits


# Decode binary chromosome and repair invalid tours
def decode_and_repair(chromosome, cities):
    values = []

    for i in range(0, len(chromosome), BITS_PER_CITY):
        bits = chromosome[i:i + BITS_PER_CITY]
        values.append(int("".join(map(str, bits)), 2))

    valid_indices = []
    seen = set()

    for value in values:
        if 0 <= value < len(cities) and value not in seen:
            valid_indices.append(value)
            seen.add(value)

    for i in range(len(cities)):
        if i not in seen:
            valid_indices.append(i)

    return [cities[i] for i in valid_indices]


# Calculate total distance including return to start
def calculate_distance(route, distances):
    total = 0

    for i in range(len(route)):
        current_city = route[i]
        next_city = route[(i + 1) % len(route)]
        total += distances[(current_city, next_city)]

    return total


# Fitness function
def fitness(chromosome, cities, distances):
    route = decode_and_repair(chromosome, cities)
    distance = calculate_distance(route, distances)
    return 1 / distance


# Generate a random valid chromosome
def create_chromosome(cities):
    indices = list(range(len(cities)))
    random.shuffle(indices)

    bits = []
    for value in indices:
        bits.extend(int(bit) for bit in format(value, "04b"))

    return bits


# Tournament selection
def tournament_selection(population, cities, distances):
    candidates = random.sample(population, TOURNAMENT_SIZE)

    return max(
        candidates,
        key=lambda chromosome: fitness(
            chromosome, cities, distances
        )
    )


# One-point crossover
def one_point_crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)

    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]

    return child1, child2


# Bit-flip mutation
def bit_flip_mutation(chromosome):
    chromosome = chromosome[:]

    for i in range(len(chromosome)):
        if random.random() < MUTATION_PROBABILITY:
            chromosome[i] = 1 - chromosome[i]

    return chromosome


# Main genetic algorithm
def genetic_algorithm(cities, distances):
    random.seed(RANDOM_SEED)

    population = [
        create_chromosome(cities)
        for _ in range(POPULATION_SIZE)
    ]

    history = []
    start_time = time.perf_counter()

    for generation in range(GENERATIONS):

        # Sort by shortest valid tour
        population.sort(
            key=lambda chromosome: calculate_distance(
                decode_and_repair(chromosome, cities),
                distances
            )
        )

        # Preserve best two individuals
        new_population = [
            chromosome[:]
            for chromosome in population[:ELITISM_COUNT]
        ]

        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(
                population, cities, distances
            )
            parent2 = tournament_selection(
                population, cities, distances
            )

            if random.random() < CROSSOVER_PROBABILITY:
                child1, child2 = one_point_crossover(
                    parent1, parent2
                )
            else:
                child1, child2 = parent1[:], parent2[:]

            child1 = bit_flip_mutation(child1)
            child2 = bit_flip_mutation(child2)

            new_population.append(child1)

            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population

        best = min(
            population,
            key=lambda chromosome: calculate_distance(
                decode_and_repair(chromosome, cities),
                distances
            )
        )

        best_route = decode_and_repair(best, cities)
        best_distance = calculate_distance(best_route, distances)
        history.append(best_distance)

        if (generation + 1) % 50 == 0:
            print(
                f"Generation {generation + 1}: "
                f"Best distance = {best_distance}"
            )

    elapsed = time.perf_counter() - start_time

    best = min(
        population,
        key=lambda chromosome: calculate_distance(
            decode_and_repair(chromosome, cities),
            distances
        )
    )

    best_route = decode_and_repair(best, cities)
    best_distance = calculate_distance(best_route, distances)

    return best_route, best_distance, history, elapsed


# Draw convergence graph
def plot_convergence(history):
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(history) + 1), history)
    plt.xlabel("Generation")
    plt.ylabel("Best tour distance")
    plt.title("TSP Genetic Algorithm Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tsp_convergence.png")
    plt.show()


# Draw the best route
def plot_route(route, coordinates):
    points = [coordinates[city] for city in route]
    points.append(points[0])

    x = [point[0] for point in points]
    y = [point[1] for point in points]

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker="o")

    for city in route:
        cx, cy = coordinates[city]
        plt.annotate(str(city), (cx, cy))

    plt.title("Best TSP Route - 15 Cities")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tsp_best_route.png")
    plt.show()


def main():
    cities, distances, coordinates = load_dataset()

    print("TSP Genetic Algorithm")
    print("Dataset: TSPLIB a280")
    print("Selected cities:", len(cities))
    print("Population:", POPULATION_SIZE)
    print("Generations:", GENERATIONS)
    print("Crossover: One-point crossover")
    print("Mutation: Bit-flip mutation")

    route, distance, history, elapsed = genetic_algorithm(
        cities, distances
    )

    valid = (
        len(route) == NUM_CITIES
        and len(set(route)) == NUM_CITIES
        and set(route) == set(cities)
    )

    print("\n--- FINAL RESULTS ---")
    print("Best route:", route + [route[0]])
    print("Best tour distance:", distance)
    print("Fitness:", 1 / distance)
    print(f"Runtime: {elapsed:.2f} seconds")
    print("Every city visited exactly once:", valid)

    plot_convergence(history)
    plot_route(route, coordinates)


if __name__ == "__main__":
    main()
