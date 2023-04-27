from GillespieParent import *
from MarkovParent import *
from MarkovModel import *
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
                parameters.append(random.uniform(0, 0.01))
            else:
                parameters.append(random.uniform(0.01, 1))
        population.append(parameters)
    return population

def identify_max_parameter_range(alpha, beta, concentration):
    '''
    Binding rates of activated proteins are restricted by the maximum activated protein concentration in the system.
    :param concentration: concentration of dox or aba
    :param alpha: maximal binding rate of protein
    :param beta: maximal unbinding rate of protein
    :return: maximum binding rate for markov chain
    '''
    max_conc_active_protein = alpha/beta * (concentration ** 2)
    max_rate = 0.5/max_conc_active_protein
    return max_rate, max_conc_active_protein

def generate_parameters_markov_parent(dox, aba, index=-1):
    ''' For Reference:
    doxf=params[0],
    doxr=params[1],
    abaf=params[2],
    abar=params[3]
    k1f_wb=params[4],
    k1r_wb=params[5],
    k2f_wb=params[6],
    k1f_phic31=params[7],
    k1r_phic31=params[8],
    k2f_phic31=params[9],
    '''
    max_rate_dox, max_wb = identify_max_parameter_range(1 + 10 ** 7, 0.01, dox)
    max_rate_aba, max_phic31 = identify_max_parameter_range(1 + 10 ** 7, 0.01, aba)
    new_params = []
    # dox_f
    new_params.append(random.uniform(0.1, 10) * (10 ** 5))
    # dox_r
    new_params.append(random.uniform(0.1, 10))
    # aba_f
    new_params.append(random.uniform(0.1, 10) * (10 ** 5))
    # aba_r
    new_params.append(random.uniform(0.01, 10))
    # k1f_wb
    k1f_wb = random.uniform(0.1 * max_rate_dox, max_rate_dox)
    k2f_wb = random.uniform(0.1 * max_rate_dox, max_rate_dox)
    k1f_phic31 = random.uniform(0.1 * max_rate_aba, max_rate_aba)
    k2f_phic31 = random.uniform(0.1 * max_rate_aba, max_rate_aba)
    new_params.append(k1f_wb)
    # k1r_wb
    new_params.append(random.uniform(0, 1-k2f_wb*max_wb))
    # k2f_wb
    new_params.append(k2f_wb)
    # k1f_phic31
    new_params.append(k1f_phic31)
    # k1r_phic31
    new_params.append(random.uniform(0, 1-k2f_phic31*max_phic31))
    # k2f_phic31
    new_params.append(k2f_phic31)
    if index > -1:
        return new_params[index]
    return new_params

def generate_initial_parameters_markov(dox, aba, population_size):
    # Generate an initial population of solutions, where each solution is a set of randomly generated parameters
    population = []
    for i in range(population_size):
        parameters = generate_parameters_markov_parent(dox, aba)
        population.append(parameters)
    return population


def select_parents(population, percentage):
    # Select the best-performing solutions from the population to be parents for the next generation

    sorted_population = sorted(population, key=lambda RecombinaseParent: RecombinaseParent.GetFitness())
    # Selecting the top percentage of parents
    parent_size = int(len(population) * percentage)
    parents = sorted_population[:parent_size]
    return parents


def crossover_and_mutate(num_children, parents, dox, aba, mutation_rate):
    # Combine the selected parents to create new offspring solutions
    offspring = []

    for _ in range(num_children):
        parents = np.random.choice(parents, size=2, replace=False)
        parent1_parameters = parents[0].GetParameters()
        parent2_parameters = parents[1].GetParameters()
        child_parameters = [parent1_parameters[j] if random.random() < 0.5 else parent2_parameters[j] for j in
                            range(len(parent1_parameters))]
        if mutation_rate > 0:
            for i in range(len(child_parameters)):
                if random.random() < mutation_rate:
                    child_parameters[i] = generate_parameters_markov_parent(dox, aba, i)
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
       # generation.append(GillespieParent(num_cells, dox, aba, parameters[i], true_values, num_processes=num_processes))

        generation.append(MarkovParent(num_cells, dox, aba, parameters[i], true_values))
    best_parents = select_parents(generation, threshold)
    for parent in best_parents:
        print(parent.GetParameters())
    children = crossover_and_mutate(num_per_generation, best_parents, dox, aba, mutation_rate=0.2)
    return best_parents, children

def RunTree(num_per_generation, num_processes, max_iterations, threshold, dox, aba, true_values, num_cells=1000, WRITE_OUT=False):
    if WRITE_OUT:
        a = open('output.txt', 'w+')
        a.close()

    parameters = generate_initial_parameters_markov(dox, aba, num_per_generation)
    best_parent_per_generation = []
    avg_fitness_per_generation = []
    time_per_generation = []
    fitness_best_per_generation = []
    best_simulation = None
    for i in range(max_iterations):
        start_time = time.time()
        best_parents, children_params = StepGeneration(num_cells, num_per_generation, num_processes, threshold, dox, aba, parameters, true_values)
        best_parent_i = f'Best Parameters: {best_parents[0].GetParameters()} \n'
        fitness_best_parent_i = f'Fitness for Best Parent: {best_parents[0].GetFitness()} \n'
        avg_fitness_i = f'Average Fitness: {sum(i.GetFitness() for i in best_parents) / len(best_parents)} \n'
        best_simulation = best_parents[0].GetSimulation()
        if best_parents[0].GetFitness() < 0.01:
            break
        parameters = children_params
        time_taken = f'Generation {i} --- {(time.time() - start_time)}s seconds --- \n'
        print(time_taken)
        time_per_generation.append(time_taken)
        if WRITE_OUT:
            with open('output.txt', 'a+') as out:
                out.write(f'{time_taken}{best_parent_i}{fitness_best_parent_i}{avg_fitness_i}')
        best_parent_per_generation.append(best_parent_i)
        avg_fitness_per_generation.append(avg_fitness_i)
        fitness_best_per_generation.append(fitness_best_parent_i)
    return best_simulation, best_parent_per_generation, fitness_best_per_generation, avg_fitness_per_generation

if __name__ == '__main__':
    num_per_generation = int(sys.argv[1])
    num_processes = int(sys.argv[2])
    max_iterations = int(sys.argv[3])
    threshold = float(sys.argv[4])
    dox = int(sys.argv[5])
    aba = int(sys.argv[6])
    true_values = [float(sys.argv[7]), float(sys.argv[8])]
    best_params, best_parents, best_parents_fitness, avg_fitness, = RunTree(num_per_generation,
                                                                    num_processes,
                                                                    max_iterations,
                                                                    threshold,
                                                                    dox,
                                                                    aba,
                                                                    true_values,
                                                                    num_cells=1000,
                                                                    WRITE_OUT=True)
