from RecombinaseParent import *
from model import *
import sys

def generate_initial_parameters(population_size, parameter_size):
    # Generate an initial population of solutions, where each solution is a set of randomly generated parameters
    ''' For Reference:
    dox_binding=params[0],
    dox_unbinding=params[1],
    aba_binding=params[2],
    aba_unbinding=params[3],
    TF_binding=params[4],
    TF_unbinding=params[5],
    WB_binding=params[6],
    WB_unbinding=params[7],
    phiC31_binding=params[8],
    phiC31_unbinding=params[9]
    '''
    population = []
    for i in range(population_size):
        parameters = []
        for j in range(parameter_size):
            if j % 2 != 0:
                parameters.append(random.uniform(10, 100))
            else:
                parameters.append(random.uniform(0, 10))
        population.append(parameters)
    return population


def select_parents(population, percentage):
    # Select the best-performing solutions from the population to be parents for the next generation

    sorted_population = sorted(population, key=lambda RecombinaseParent: RecombinaseParent.GetFitness())
    # Selecting the top percentage of parents
    parent_size = int(len(population) * percentage)
    parents = sorted_population[:parent_size]
    return parents


def crossover_and_mutate(num_children, parents, mutation_rate):
    # Combine the selected parents to create new offspring solutions
    offspring = []

    for _ in range(num_children):
        parent1, parent2 = random.choices(parents, k=2)
        parent1_parameters = parent1.GetParameters()
        parent2_parameters = parent2.GetParameters()
        child_parameters = [parent1_parameters[j] if random.random() < 0.5 else parent2_parameters[j] for j in
                            range(len(parent1_parameters))]
        if mutation_rate > 0:
            for i in range(len(child_parameters)):
                if random.random() < mutation_rate:
                    child_parameters[i] = child_parameters[i] * random.uniform(0.5, 1.5)
        offspring.append(child_parameters)
    return offspring

def StepGeneration(num_cells, num_per_generation, num_processes, threshold, dox, aba, parameters, true_values):
    """
    Each Generation runs and picks the next offspring.
    """
    generation = []
    # with Pool(num_processes) as p:
    #     generation = p.starmap(RecombinaseParent, list_args)
    for i in range(num_per_generation):
        generation.append(RecombinaseParent(num_cells, dox, aba, parameters[i], true_values, num_processes=num_processes))
    best_parents = select_parents(generation, threshold)
    for parent in best_parents:
        print(parent.GetParameters())
    children = crossover_and_mutate(num_per_generation, best_parents, mutation_rate=0.2)
    return best_parents, children

def RunTree(num_per_generation, num_processes, max_iterations, threshold, dox, aba, true_values, num_cells=1000, WRITE_OUT=False):
    parameters = generate_initial_parameters(num_per_generation, 10)
    best_parent_per_generation = []
    avg_fitness_per_generation = []
    time_per_generation = []
    best_simulation = None
    for i in range(max_iterations):
        start_time = time.time()
        best_parents, children_params = StepGeneration(num_cells, num_per_generation, num_processes, threshold, dox, aba, parameters, true_values)
        best_parent_per_generation.append(f'Best Parameters: {best_parents[0].GetParameters()} \n')
        avg_fitness_per_generation.append(f'Average Fitness: {sum(i.GetFitness() for i in best_parents) / len(best_parents)} \n')
        best_simulation = best_parents[0].GetSimulation()
        if best_parents[0].GetFitness() < 0.1:
            break
        parameters = children_params
        time_taken = f'Generation {i} --- {(time.time() - start_time)}s seconds --- \n'
        print(time_taken)
        time_per_generation.append([time_taken])
    if WRITE_OUT:
        with open('output.txt', 'w') as out:
            for s, best, avg in zip(time_per_generation, best_parent_per_generation, avg_fitness_per_generation):
                out.write(f'{s}{best}{avg}')
    return best_simulation, best_parent_per_generation, avg_fitness_per_generation

if __name__ == '__main__':
    num_per_generation = int(sys.argv[1])
    num_processes = int(sys.argv[2])
    max_iterations = int(sys.argv[3])
    threshold = float(sys.argv[4])
    dox = int(sys.argv[5])
    aba = int(sys.argv[6])
    true_values = [float(sys.argv[7]), float(sys.argv[8])]
    best_params, best_parents, avg_fitness = RunTree(num_per_generation, num_processes, max_iterations, threshold, dox, aba, true_values)