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
        gene = {
            'genotype': genotype,
            'phenotype': genotype,
            'evaluation': 0
        }
        return gene

    def generate_valid_gene(self, number_of_items):
        gene = []
        for x in range(0, number_of_items):
            # value = random.uniform(min_value, max_value)
            value = random.randint(0, 1)
            gene.append(value)
        return gene

    def generate_random_population(self, size_of_pop, number_of_features):
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
        for gene in population:
            selected_features = self.get_selected_features(gene)
            model = learner.learn(selected_features, data_training)
            score = learner.evaluate(model, selected_features, data_test)

            gene["evaluation"] = score

    def get_max_gene(self, population):
        max_value = -sys.maxsize
        max_gene = None
        for gene in population:
            if gene['evaluation'] > max_value:
                max_value = gene['evaluation']
                max_gene = gene
        return max_gene

    def pick_parents(self, population):
        """
        Tournament Selection
        :param population:
        :return:
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
        if random.random() < mutation_rate:
            position = random.randint(0, len(genotype)-1)
            genotype[position] = random.randint(0, 1)
        return genotype

    # </editor-fold>

    # <editor-fold desc="Helpers">
    def get_selected_features(self, gene):
        selected_features = []
        genotype = gene["genotype"]
        for index in range(len(genotype)):
            if genotype[index] == 1:
                selected_features.append(index)
        return selected_features

    def print_gene(self, gene, generation, should_print):
        if not should_print:
            return
        print("Generation: {}".format(generation))
        print("Genotype of Max: {}".format(gene["genotype"]))
        print("Phenotype of Max: {}".format(gene["phenotype"]))
        print("Fitness of Max: {}".format(gene["evaluation"]))
        return
    # </editor-fold>


# all_data = CustomCSVReader.read_file("data/iris.data.txt", float)
# data_training = all_data[:2*int(len(all_data)/3)]
# data_test = all_data[2*int(len(all_data)/3):]
# feature_length = len(all_data[0]) - 1
#
# testLearner = TestLeaner()
#
# GA = GeneticAlgorithmFeatureSelection()
# best_features = GA.select_features_ga(testLearner, data_training, data_test, feature_length, number_of_generations=5)
#
# print("  ")
# print("FINAL!!!!")
# print("Genotype of Max: {}".format(best_features["genotype"]))
# print("Phenotype of Max: {}".format(best_features["phenotype"]))
# print("Fitness of Max: {}".format(best_features["evaluation"]))

def run_ga_kmeans_experiment(data_set_path, number_of_clusters, learner, fraction_of_data_used=1, data_type=float):
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



# sys.stdout = open('results/GA-Kmeans-iris-results.txt', 'w')
run_ga_kmeans_experiment("data/iris.data.txt", 3, KMeans(3))

# sys.stdout = open('results/GA-Kmeans-glass-results.txt', 'w')
# run_ga_kmeans_experiment("data/glass.data.txt", 6, KMeans(6))
#
# sys.stdout = open('results/GA-Kmeans-spambase-results.txt', 'w')
# run_ga_kmeans_experiment("data/spambase.data.txt", 2, KMeans(2))