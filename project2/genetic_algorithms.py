import sys
import random
import pprint
from testLearner import TestLeaner
from customCsvReader import CustomCSVReader
from k_means import KMeans
from hierarchical_agglomerative_clustering import HAC


class GeneticAlgorithmFeatureSelection:
    # <editor-fold desc="Init Population">
    def create_gene(self, genotype):
        """
        Create gene with the below structure
        :param genotype: The genotype to use
        :return: dict, gene
        """
        gene = {
            'genotype': genotype,
            'phenotype': genotype,
            'evaluation': 0
        }
        return gene

    def generate_valid_gene(self, number_of_items):
        """
        generates valid genes.
        Geno type is a 1's hot encoding for features.
        :param number_of_items: int, number of items in a gene
        :return: gene.
        """
        gene = []
        for x in range(0, number_of_items):
            # value = random.uniform(min_value, max_value)
            value = random.randint(0, 1)
            gene.append(value)
        return gene

    def generate_random_population(self, size_of_pop, number_of_features):
        """
        Gerneates a random population to start the GA with.
        :param size_of_pop: int
        :param number_of_features: int, how big the gense should be.
        :return: list of genes, population
        """
        population = []
        for x in range(size_of_pop):
            genotype = self.generate_valid_gene(number_of_features)
            gene = {
                'genotype': genotype,
                'phenotype': genotype,
                'evaluation': 0
            }
            population.append(gene)
        return population
    # </editor-fold>

    # <editor-fold desc="Main">
    def select_features_ga(self, learner, data_training, data_test, feature_length, population_size=25, crossover_rate=.8,
                           mutation_rate=.6, number_of_generations=20):
        """
        This is the main work horse of the GA.
        Works by evaluating a population, then creating the next generation and so on until all generations have been
        completed.
        It uses the learner to learn the model that we are using for selection, and then it uses the evaluation of that
        learner to get the fitness of a gene.

        Uses a 1's hot encoding for the genotype and which feature is being used.

        :param learner: A clustering algorithm that has "learn" and "evaluate" functions
        :param data_training: list of list traing data
        :param data_test: list of list test data
        :param feature_length: int, length of a given data point, minus the class
        :param population_size: int
        :param crossover_rate: float
        :param mutation_rate: float
        :param number_of_generations: int
        :return: Best Feature found
        """

        population = self.generate_random_population(population_size, feature_length)
        self.evaluate_population(population, learner, data_training, data_test)

        best_all_time = self.get_max_gene(population)

        for generation in range(0, number_of_generations):
            next_population = []
            for n in range(0, int(len(population) / 2)):
                parents = self.pick_parents(population)
                children = self.reproduce(parents[0]["genotype"], parents[1]["genotype"], feature_length,
                                          crossover_rate)

                child1 = self.mutate(children[0], mutation_rate)
                child2 = self.mutate(children[1], mutation_rate)

                child1 = self.create_gene(child1)
                child2 = self.create_gene(child2)

                next_population.append(child1)
                next_population.append(child2)
            population = next_population
            self.evaluate_population(population, learner, data_training, data_test)

            max_gene = self.get_max_gene(population)
            self.print_gene(max_gene, generation, True)
            best_all_time = self.get_max_gene((best_all_time, max_gene))

        return best_all_time

    # </editor-fold>

    # <editor-fold desc="Population Manipulation">
    def evaluate_population(self, population, learner, data_training, data_test):
        """
        Evaluates a population.
        Uses the LEARNER to learn based on training data, and evaluates with test data.
        :param population: list of genes, population
        :param learner: Clustering algorithm being used.
        :param data_training: list of list
        :param data_test: list of list
        :return: gene with evaluation
        """
        for gene in population:
            selected_features = self.get_selected_features(gene)
            model = learner.learn(selected_features, data_training)
            score = learner.evaluate(model, selected_features, data_test)

            gene["evaluation"] = score

    def get_max_gene(self, population):
        """
        Gets the gene with the best fitness.
        :param population: list of list, pop
        :return: gene with max score.
        """
        max_value = -sys.maxsize
        max_gene = None
        for gene in population:
            if gene['evaluation'] > max_value:
                max_value = gene['evaluation']
                max_gene = gene
        return max_gene

    def pick_parents(self, population):
        """
        Tournament Selection to pick two parent genes.
        uses a tournament of 7 genes and then picks the best of those.
        :param population: list of list, pop
        :return: tuple, two genes
        """
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

            max_score = -sys.maxsize
            max_parent = None
            for parent in possible_parents:
                if parent['evaluation'] >= max_score:
                    max_parent = parent
                    max_score = parent['evaluation']

            parents.append(max_parent)

        return tuple(parents)
    # </editor-fold>

    # <editor-fold desc="Children">
    def reproduce(self, parent1_genotype, parent2_genotype, crossover_len, crossover_rate):
        """
        A function that does the crossover part of GA
        :param parent1_genotype: list a genotype
        :param parent2_genotype: list a genotyp
        :param crossover_len: length of gene
        :param crossover_rate: probability of crossover happening
        :return: a tuple of children
        """
        children = ()
        if random.random() < crossover_rate:
            crossover_point = random.randint(0, crossover_len)
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

    def mutate(self, genotype, mutation_rate):
        """
        Performs the mutate function for GA
        :param genotype: list genotype
        :param mutation_rate: float, rate of mutation
        :return: mutated genotype.
        """
        if random.random() < mutation_rate:
            position = random.randint(0, len(genotype)-1)
            genotype[position] = random.randint(0, 1)
        return genotype

    # </editor-fold>

    # <editor-fold desc="Helpers">
    def get_selected_features(self, gene):
        """
        Changes the 1's hot encoding of the gene to a list of selected features, 0 indexed.
        :param gene: dict as described by create gene
        :return: the selected features 0 indexed in an array
        """
        selected_features = []
        genotype = gene["genotype"]
        for index in range(len(genotype)):
            if genotype[index] == 1:
                selected_features.append(index)
        return selected_features

    def print_gene(self, gene, generation, should_print):
        """
        A debug function to print the gene
        :param gene: dict as described in create gene
        :param generation: int, generation number
        :param should_print: boolean
        :return: nothing
        """
        if not should_print:
            return
        print("Generation: {}".format(generation))
        print("Genotype of Max: {}".format(gene["genotype"]))
        print("Phenotype of Max: {}".format(gene["phenotype"]))
        print("Fitness of Max: {}".format(gene["evaluation"]))
        return
    # </editor-fold>

def run_ga_kmeans_experiment(data_set_path, number_of_clusters, learner, fraction_of_data_used=1, data_type=float):
    """
    The main work horse for running the experiments and output the approriate information into a file

    Works by reading in the data, trainnig and test data.

    Creates the GA and pass the needed data to it. It returns a list of selected features.
    The mean is then retrieved for those features, and we cluster all of the data based on the means

    Finally, I print all information needed in a human readable way.
    """
    print("Running {0} Experiment with k clusters = {1}".format(data_set_path, number_of_clusters))
    all_data = CustomCSVReader.read_file(data_set_path, data_type)
    feature_selection_data = all_data[:int(len(all_data)/fraction_of_data_used)]
    feature_length = len(all_data[0]) - 1

    Features = list(range(feature_length))
    GA = GeneticAlgorithmFeatureSelection()
    best_features = GA.select_features_ga(learner, feature_selection_data, all_data, feature_length)

    # feature_ones_hot = best_features["genotype"]
    selected_features = GA.get_selected_features(best_features)

    means = learner.learn(selected_features, all_data)
    data_clusters = learner.get_clusters_for_means(means, selected_features, all_data)

    print("The Final Selected Features are: (features are zero indexed) ")
    print("{}\n".format(selected_features))
    print("The Fisher Score for the clustering is: ")
    print("{}\n".format(best_features["evaluation"]))

    pp = pprint.PrettyPrinter(indent=2, width=400)
    print("For Clustered points, the key in the dictionary represents the cluster each data point belongs to. ")
    print("Clustered points: ")
    pp.pprint(data_clusters)


def run_hac_experiment(data_set_path, number_of_clusters, hac, fraction_of_data_used=1, data_type=float):
    """
    The main work horse for running the experiments and output the approriate information into a file

    Works by reading in the data, trainnig and test data.

    Creates the HAC and pass the needed data to it. It returns a list of selected features.
    The cluster of datapoints by HAC is then retrieved for those features on the test data. (Cluster if datapoint Ids = model)
    We then retrieve the full clustering of the data from HAC by passing in the "model" it returned.

    Results in all of the datapoint being clustered by HAC

    Finally, I print all information needed in a human readable way.
    """
    print("Running {0} Experiment with k clusters = {1}".format(data_set_path, number_of_clusters))
    all_data = CustomCSVReader.read_file(data_set_path, data_type)
    feature_selection_data = all_data[:int(len(all_data)/fraction_of_data_used)]
    feature_length = len(all_data[0]) - 1

    Features = list(range(feature_length))
    GA = GeneticAlgorithmFeatureSelection()
    best_features = GA.select_features_ga(hac, feature_selection_data, all_data, feature_length)

    selected_features = GA.get_selected_features(best_features)

    clusters_of_datapoint_ids = hac.learn(selected_features, feature_selection_data)
    full_clusters = hac.get_full_clusters_of_data(clusters_of_datapoint_ids, selected_features, all_data)

    print("The Final Selected Features are: (features are zero indexed) ")
    print("{}\n".format(selected_features))
    print("The Fisher Score for the clustering is: ")
    print("{}\n".format(best_features["evaluation"]))

    pp = pprint.PrettyPrinter(indent=2, width=400)
    print("For Clustered points, the key in the dictionary represents the cluster each data point belongs to. ")
    print("Clustered points: ")
    pp.pprint(full_clusters)


# KMeans experiments
sys.stdout = open('results/GA-Kmeans-iris-results.txt', 'w')
run_ga_kmeans_experiment("data/iris.data.txt", 3, KMeans(3))

sys.stdout = open('results/GA-Kmeans-glass-results.txt', 'w')
run_ga_kmeans_experiment("data/glass.data.txt", 6, KMeans(6))

sys.stdout = open('results/GA-Kmeans-spambase-results.txt', 'w')
run_ga_kmeans_experiment("data/spambase.data.txt", 2, KMeans(2), fraction_of_data_used=100)

# HAC experiments
sys.stdout = open('results/GA-HAC-iris-results.txt', 'w')
run_hac_experiment("data/iris.data.txt", 3, HAC(3))

sys.stdout = open('results/GA-HAC-glass-results.txt', 'w')
run_hac_experiment("data/glass.data.txt", 6, HAC(6))

sys.stdout = open('results/GA-HAC-spambase-results.txt', 'w')
run_hac_experiment("data/spambase.data.txt", 2, HAC(2), fraction_of_data_used=100)

