import sys, random


def sphere( shift, xs):
    return sum( [(x - shift)**2 for x in xs])

parameters = {
   "f": lambda xs: sphere( 0.5, xs),
   "minimization": True
   # put other parameters in here.
}

####################################################

def pad_zeros(bin_str, number_of_binary_digits_total):
    for x in range(10 - len(bin_str)): 
        bin_str = '0' + bin_str
    return bin_str

def bin_str_to_int(str_value):
    return int(str_value, 2)

def int_to_bin_str(int_value):
    return pad_zeros(bin(int_value)[2:], 10)

def decode_binary_to_512_range(str_value):
    binary_value = int(str_value, 2) 
    decoded_value = (binary_value - 512)/100.0
    return decoded_value

def encode_512_range_to_binary(float_value):
    adjusted_value = (float_value*100) + 512
    adjusted_value = int(adjusted_value)
    bin_str = int_to_bin_str(adjusted_value)
    bin_str = pad_zeros(bin_str, 10)
    return bin_str


def perform_mutate_binary(str_value, position):
    mutated_gene = ''
    for x in range(0, len(str_value)):
        if x == position: 
            if str_value[x] == '1':
                mutated_gene = mutated_gene + '0'
            else: 
                mutated_gene = mutated_gene + '1'
        else:
            mutated_gene = mutated_gene + str_value[x]
    return mutated_gene

def mutate_binary(str_value, mutate_rate):
    if random.random() < mutate_rate: 
        position = random.randint(0, 100)
        return perform_mutate_binary(str_value, position)
    return str_value



def get_phenotype_bin(genotype_str):
    phenotype = []
    for x in range(0,10):
        genotype_sub = genotype_str[10*x : 10*(x+1)]
        geno_value = decode_binary_to_512_range(genotype_sub)
        phenotype.append(geno_value)
    return phenotype

def generate_valid_gene(max_value, number_of_items):
    gene = ''
    for x in range(0, number_of_items):
        value = random.randint(0, max_value)
        gene = gene + int_to_bin_str(value) 

    return gene

def generate_valid_gene_real(min_value, max_value, number_of_items):
    gene = []
    for x in range(0, number_of_items):
        value = random.uniform(min_value, max_value)
        gene.append(value)
    return gene


def generate_random_population_binary(size_of_pop):
    population = []
    for x in range(size_of_pop):
        genotype = generate_valid_gene(1023, 10)
        phenotype = get_phenotype_bin(genotype)
        gene = {
            'genotype': genotype,
            'phenotype': phenotype,
            'evaluation': 0
        }
        population.append(gene)
    return population

def create_gene_binary(genotype):
    phenotype = get_phenotype_bin(genotype)
    gene = {
        'genotype': genotype,
        'phenotype': phenotype,
        'evaluation': 0
    }
    return gene

def create_gene_real(genotype):
    # phenotype = get_phenotype_bin(genotype)
    gene = {
        'genotype': genotype,
        'phenotype': genotype,
        'evaluation': 0
    }
    return gene

""" 
Using tournament style selection
"""
def pick_parents(population):
    
    parents = []
    for num_par in range(0,2):
        random_index_selection = []
        for x in range(0, 7):
            index = random.randint(0, len(population)-1)
            while index in random_index_selection:
                index = random.randint(0, len(population)-1)
            random_index_selection.append(index)
        
        # print( "Random index_selection: {}".format(random_index_selection)
        
        possible_parents = []
        for index in random_index_selection:
            possible_parents.append(population[index])
        
        max_score = -sys.maxint
        max_parent = None
        for parent in possible_parents:
            if parent['evaluation'] >= max_score:
                max_parent = parent
                max_score = parent['evaluation']
        
        parents.append(max_parent)
    
    return tuple(parents)


def reproduce(parent1_genotype, parent2_genotype, crossover_len, crossover_rate): 
    children = ()
    if random.random() < crossover_rate: 
        crossover_point = random.randint(0,crossover_len)
        parent1_part1 = parent1_genotype[0:crossover_point]
        parent1_part2 = parent1_genotype[crossover_point:]
        parent2_part1 = parent2_genotype[0:crossover_point]
        parent2_part2 = parent2_genotype[crossover_point:]

        child1 = parent1_part1 + parent2_part2
        child2 = parent2_part1 + parent1_part2 

        children = (child1, child2) 
    else: 
        children = (parent1_genotype, parent2_genotype)
    return children

def is_valid_gene(phenotype):
    for value in phenotype: 
        if value < -5.12 or value > 5.12: 
            return False
    return True

def evaluate_population(population, fitness_function, minimize):
    for gene in population: 
        score = 0
        if is_valid_gene(gene["phenotype"]):
            score = fitness_function(gene["phenotype"])
        else: 
            score = -sys.maxint

        if minimize: 
            score = 1/(1+ score)
        
        gene["evaluation"] = score

def get_max_gene(population):
    max_value = -sys.maxint
    max_gene = None
    for gene in population: 
        if gene['evaluation'] > max_value:
            max_value = gene['evaluation']
            max_gene = gene
    return max_gene


def perform_mutate_real(genotype, position, standard_deviation):
    devation = random.gauss(mu = 0, sigma = standard_deviation)
    genotype[position] = genotype[position] + devation
    return genotype


def mutate_real(genotype, mutation_rate, standard_deviation):
    if random.random() < mutation_rate: 
        position = random.randint(0, 9)
        return perform_mutate_real(genotype, position, standard_deviation)
    return genotype


def print_gene(gene, generation, parameters, should_print):
    if not should_print:
        return
    print( "Generation: {}".format(generation))
    print( "Genotype of Max: {}".format(gene["genotype"]))
    print( "Phenotype of Max: {}".format(gene["phenotype"]))
    print( "Fitness of Max: {}".format(gene["evaluation"]))
    print( "Function Value: {}".format(parameters["f"](gene["phenotype"])))


def binary_ga( parameters):
    population = generate_random_population_binary(parameters["population_size"])
    evaluate_population(population, parameters["f"], parameters["minimization"])
    best_all_time = get_max_gene(population)
    for generation in range(0, parameters["num_gens"]):
        next_population = []
        for n in range(0, len(population)/2):
            parents = pick_parents(population)
            children = reproduce(parents[0]["genotype"], parents[1]["genotype"], 100, parameters["crossover_rate"])
            
            child1 = mutate_binary(children[0], parameters["mutation_rate"])
            child2 = mutate_binary(children[1], parameters["mutation_rate"])
            
            child1 = create_gene_binary(child1)
            child2 = create_gene_binary(child2)

            next_population.append(child1)
            next_population.append(child2)
        population = next_population
        evaluate_population(population, parameters["f"], parameters["minimization"])
        
        max_gene = get_max_gene(population)
        if generation % 20 == 0:
            print_gene(max_gene, generation, parameters, parameters['DEBUG'])
        best_all_time = get_max_gene((best_all_time, max_gene))
    return get_max_gene(population)


def generate_random_population_real(size_of_pop):
    population = []
    for x in range(size_of_pop):
        genotype = generate_valid_gene_real(-5.12, 5.12, 10)
        gene = {
            'genotype': genotype,
            'phenotype': genotype,
            'evaluation': 0
        }
        population.append(gene)
    return population


def real_ga( parameters):
    population = generate_random_population_real(parameters["population_size"])
    evaluate_population(population, parameters["f"], parameters["minimization"])
    best_all_time = get_max_gene(population)
    for generation in range(0, parameters["num_gens"]):
        next_population = []
        for n in range(0, len(population)/2):
            parents = pick_parents(population)
            children = reproduce(parents[0]["genotype"], parents[1]["genotype"], 10, parameters["crossover_rate"])
            
            child1 = mutate_real(children[0], parameters["mutation_rate"], parameters["standard_deviation"])
            child2 = mutate_real(children[1], parameters["mutation_rate"], parameters["standard_deviation"])
            
            child1 = create_gene_real(child1)
            child2 = create_gene_real(child2)

            next_population.append(child1)
            next_population.append(child2)
        population = next_population
        evaluate_population(population, parameters["f"], parameters["minimization"])

        max_gene = get_max_gene(population)
        print_gene(max_gene, generation, parameters, parameters['DEBUG'])
        best_all_time = get_max_gene((best_all_time, max_gene))

    return best_all_time



parameters = {
   "f": lambda xs: sphere( 0.5, xs),
   "minimization": True,
   # put other parameters in here.
   "DEBUG": True, 
   "num_gens": 1000,
   "mutation_rate": .6, 
   "crossover_rate": 1,
   "population_size": 500,
#    "within_amount": .3
}


max_gene = binary_ga(parameters)

print("  ")
print("FINAL!!!!")
print( "Genotype of Max: {}".format(max_gene["genotype"]))
print( "Phenotype of Max: {}".format(max_gene["phenotype"]))
print( "Fitness of Max: {}".format(max_gene["evaluation"]))
print( "Function Value: {}".format(parameters["f"](max_gene["phenotype"])))

parameters = {
   "f": lambda xs: sphere( 0.5, xs),
   "minimization": False,
   # put other parameters in here.
   "DEBUG": True, 
   "num_gens": 100,
   "mutation_rate": .6, 
   "crossover_rate": 1,
   "population_size": 500,
#    "within_amount": .3
    "standard_deviation": .2
}


# max_gene = real_ga(parameters)
#
# print( "  ")
# print( "FINAL!!!!")
# print( "Genotype of Max: {}".format(max_gene["genotype"]))
# print( "Phenotype of Max: {}".format(max_gene["phenotype"]))
# print( "Fitness of Max: {}".format(max_gene["evaluation"]))
# print( "Function Value: {}".format(parameters["f"](max_gene["phenotype"])))
