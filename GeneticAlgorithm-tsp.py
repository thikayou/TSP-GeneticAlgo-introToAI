import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import time

# Load cities from a text file
def getCity(filename='cities.txt'):
    cities = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            city_number = int(parts[0])
            x_coord = float(parts[1])
            y_coord = float(parts[2])
            cities.append([city_number, x_coord, y_coord])
    return cities


# Calculate distance of the cities
def evaluate_fitness(cities):
    total_sum = 0
    for i in range(len(cities) - 1):
        cityA = cities[i]
        cityB = cities[i + 1]
        d = math.sqrt(math.pow(cityB[1] - cityA[1], 2) + math.pow(cityB[2] - cityA[2], 2))
        total_sum += d
    cityA = cities[0]
    cityB = cities[-1]
    d = math.sqrt(math.pow(cityB[1] - cityA[1], 2) + math.pow(cityB[2] - cityA[2], 2))
    total_sum += d
    return total_sum


# Select the population
def selectPopulation(cities, size):
    population = []
    for i in range(size):
        c = cities.copy()
        random.shuffle(c)  # Reorganize the order of cities
        distance = evaluate_fitness(c)  # Calculate total distance
        population.append([distance, c])  # Add to population
    fittest = sorted(population)[0]  # Get the shortest distance
    return population, fittest


# Genetic algorithm
def geneticAlgorithm(
    population,
    lenCities,
    GENERATION,
    TOURNAMENT_SELECTION_SIZE,
    MUTATION_RATE,
    CROSSOVER_RATE,
):
    start_time = time.time()
    fitness_history = []  # Track best fitness over generations
    best_fitness_history = []
    gen_number = 0
    place_holder = []
    all_fitness_history = []
    fitness_history.append([individual[0] for individual in population])
    best_distance = float('inf')  # Initialize best distance
    target = float('inf')  # Initialize target

    for i in range(GENERATION):

        new_population = []
             # selecting two of the best options we have (elitism)
        new_population.append(sorted(population)[0])
        new_population.append(sorted(population)[1])
        
        fitness_values = [ind[0] for ind in population]
        all_fitness_history.append(fitness_values)  # Store all fitness values

        for i in range(int((len(population) - 2) / 2)):
            # CROSSOVER
            random_number = random.random()
            if random_number < CROSSOVER_RATE:
                parent_chromosome1 = sorted(random.choices(population, k=TOURNAMENT_SELECTION_SIZE))[0]
                parent_chromosome2 = sorted(random.choices(population, k=TOURNAMENT_SELECTION_SIZE))[0]
                point = random.randint(0, lenCities - 1)
                child_chromosome1 = parent_chromosome1[1][0:point]
                for j in parent_chromosome2[1]:
                    if j not in child_chromosome1:
                        child_chromosome1.append(j)

                child_chromosome2 = parent_chromosome2[1][0:point]
                for j in parent_chromosome1[1]:
                    if j not in child_chromosome2:
                        child_chromosome2.append(j)

            # If crossover not happen
            else:
                child_chromosome1 = random.choices(population)[0][1]
                child_chromosome2 = random.choices(population)[0][1]

             # MUTATION
            if random.random() < MUTATION_RATE:
                point1 = random.randint(0, len(child_chromosome1) - 1)
                point2 = random.randint(0, len(child_chromosome1) - 1)
                child_chromosome1[point1], child_chromosome1[point2] = (
                    child_chromosome1[point2],
                    child_chromosome1[point1],
                )
                point1 = random.randint(0, len(child_chromosome2) - 1)
                point2 = random.randint(0, len(child_chromosome2) - 1)
                child_chromosome2[point1], child_chromosome2[point2] = (
                    child_chromosome2[point2],
                    child_chromosome2[point1],
                )

            new_population.append([evaluate_fitness(child_chromosome1), child_chromosome1])
            new_population.append([evaluate_fitness(child_chromosome2), child_chromosome2])

        previous_best = population[0][0]
        population = new_population
        current_best = population[0][0]

        gen_number += 1
        best_fitness_history.append(current_best)

         # Print the best distance for the current 10 generation
        # if gen_number % 10 == 0:
        #     print(f"Generation {gen_number}   |   Best Distance = {current_best}")
    

        if current_best < best_distance:
            best_distance = current_best # Update best distance
            target = best_distance * 0.95  # Update the target based on best distance

        if current_best < target:
            break

        if 0 <= current_best - previous_best < previous_best * 0.01:
            place_holder.append(current_best)
            if len(place_holder) > 0.2 * GENERATION:
                break
        else:
            place_holder = []

    answer = sorted(population)[0]
    end_time = time.time()
    elapsed_time = end_time - start_time

    # print(f"Total running time: {elapsed_time:.4f} seconds")  # Print the total running time

    return answer, gen_number, best_fitness_history, fitness_history, target, elapsed_time


# Draw cities and answer map
def drawMap(city, answer):
    for j in city:
        plt.plot(j[1], j[2], "ro")
        plt.annotate(j[0], (j[1], j[2]))

    for i in range(len(answer[1])): 
        try:
            first = answer[1][i]
            second = answer[1][i + 1]
            plt.plot([first[1], second[1]], [first[2], second[2]], "gray")
        except:
            continue

    first = answer[1][0]
    second = answer[1][-1]
    plt.plot([first[1], second[1]], [first[2], second[2]], "gray")
    plt.show()


# Function to plot convergence
def plot_generations(best_fitness_history):
    generations = list(range(len(best_fitness_history)))  # Generate generation numbers
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness_history, linestyle='-', color='b')
    plt.title('GA Best Distance Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Distance')
    plt.grid(True)
    plt.xticks(range(0, len(best_fitness_history), 10))  # Show ticks at intervals of 10    
    plt.show()

# Function to get parameters based on the number of cities
def get_parameters(num_cities):
    if num_cities <= 10:
        return 200, 40, 2, 0.3, 0.9  # Smaller population, generations, tournament size
    elif num_cities <= 20:
        return 1000, 100, 3, 0.2, 0.85  # Medium settings
    else:
        return 2000, 200, 4, 0.1, 0.9  # Larger settings

def main():
    cities = getCity('cities.txt')  # Load your text file
    num_cities = len(cities)
    
    # Get parameters based on the number of cities
    POPULATION_SIZE, GENERATION, TOURNAMENT_SELECTION_SIZE, MUTATION_RATE, CROSSOVER_RATE = get_parameters(num_cities)

    iteration = 1
    for i in range(iteration):
        firstPopulation, firstFittest = selectPopulation(cities, POPULATION_SIZE)
        answer, genNumber, best_fitness_history, fitness_history, target, elapsed_time = geneticAlgorithm(
            firstPopulation,
            num_cities,
            GENERATION,
            TOURNAMENT_SELECTION_SIZE,
            MUTATION_RATE,
            CROSSOVER_RATE,
        )

        print("\n----------------------------------------------------------------")
        print("GENETIC ALGORITHM")
        print("Generation: " + str(genNumber))
        print("Fittest chromosome distance before training: " + str(firstFittest[0]))
        print("Fittest chromosome distance after training: " + str(answer[0]))
        print("Target distance: " + str(target))
        print(f"Total running time: {elapsed_time:.4f} seconds")
        print("----------------------------------------------------------------\n")

        # Extract city numbers from the best solution
        best_route = answer[1]  # Assuming answer[1] contains the best route
        city_numbers = [city[0] for city in best_route]  # Extract city numbers

        drawMap(cities, answer)
        plot_generations(best_fitness_history)

        # Save results to CSV
        pd.DataFrame({'Generation': range(len(best_fitness_history)), 'Best Distance': best_fitness_history}).to_csv('GA_best_route.csv', index=False)
        # Convert city numbers to a DataFrame with a single row
        city_numbers_df = pd.DataFrame([city_numbers])

        # Save to CSV without line breaks
        city_numbers_df.to_csv('GA_best_route_cities.csv', header=False, index=False)
        return elapsed_time

if __name__ == "__main__":
    main()