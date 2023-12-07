import matplotlib as mpl
import cProfile
import pstats
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import seaborn as sns
import pandas as pd
try:
    import itertools.izip as zip
except ImportError:
    import itertools

# This is taken from Trevor Bedford's github. It is a Wright-Fisher simulation which includes: Mutation, Selection and Genetic Drift

def main():
    
    pop_size = 10000                        # Size of Population
    seq_length = 150                         # This is "N"
    drift_generations = 1000                 # How long it runs
    drug_generations = 5
    mutation_rate = 0.000025                # per gen per individual per site
    mutation_rate_drug = 0.0
    repeats = 300                      # Number of repeats

    
    fitness_cost = 0                      # (1 - this value) gives growth of the resistant at 100% resistant at all fractions when in drug.
    growth_of_base_haplotype_in_drug = 0.01

    alphabet = ['A', 'C', 'T', 'G']
    base_haplotype = ''.join(["A" for i in range(seq_length)])
    
    histy_repeats = []
    histy_popped_up_repeats =[]
    histy_at_end_drug_repeats = []
    
    histy_fc_repeats = []
    histy_fc_popped_up_repeats =[]
    histy_fc_at_end_drug_repeats = []

    total_muts = np.zeros((repeats,drift_generations))

    fmax = []
    
    start = time.time()
    for repeat in range(repeats):
        print(repeat)

        pop = {}
        fitness = {}
        sens_fit = {}
        fitness_drug = {}
        sens_fit_drug = {}

        pop[base_haplotype] = pop_size
        fitness[base_haplotype] = 1
        sens_fit[base_haplotype] = 1
        
        fitness_drug[base_haplotype] = growth_of_base_haplotype_in_drug
        sens_fit_drug[base_haplotype] = growth_of_base_haplotype_in_drug

        history = []
        
        
        simulate(pop, history, drift_generations, mutation_rate, pop_size, seq_length, fitness, fitness_drug, sens_fit, sens_fit_drug, alphabet, base_haplotype, fitness_cost, drug=0)
        
        non_base_sens_fit = copy.deepcopy(sens_fit)
        non_base_sens_fit.pop(base_haplotype)

        fmax.append(non_base_sens_fit[max(non_base_sens_fit, key=non_base_sens_fit.get)])
  
        """
        Calc stuff for first drift drift_generations
        """
        
        haplotypes_at_end = list(pop.keys())
        histy = []
        histy_fc = []
        for haplotype in haplotypes_at_end:
            if haplotype != base_haplotype:
                for i in range(pop[haplotype]):
                    histy.append(sens_fit[haplotype])
                    histy_fc.append(fitness[haplotype])
                
        histy_repeats += histy
        histy_fc_repeats += histy_fc           
        

        haplotypes_that_occured = list(sens_fit.keys())
        histy_popped_up = []
        histy_fc_popped_up = []
        for haplotype in haplotypes_that_occured:
            if haplotype != base_haplotype:
                histy_popped_up.append(sens_fit[haplotype])
                histy_fc_popped_up.append(fitness[haplotype])
                
        histy_popped_up_repeats += histy_popped_up
        histy_fc_popped_up_repeats += histy_fc_popped_up

        
        
        
        cm = plt.cm.get_cmap('bwr')
        x_span = 1
        
        Y_histy, X_histy = np.histogram(histy, 40, density=True)
        C_histy = [cm(((x-0)/x_span)) for x in X_histy]
        
        Y_histy_popped_up, X_histy_popped_up = np.histogram(histy_popped_up, 40, density=True)
        C_histy_popped_up = [cm(((x-0)/x_span)) for x in X_histy_popped_up]
        
        
        Y_histy_fc, X_histy_fc = np.histogram(histy_fc, 40, density=True)
        C_histy_fc = [cm(((x-0)/x_span)) for x in X_histy_fc]
        
        Y_histy_fc_popped_up, X_histy_fc_popped_up = np.histogram(histy_fc_popped_up, 40, density=True)
        C_histy_fc_popped_up = [cm(((x-0)/x_span)) for x in X_histy_fc_popped_up]
        
        """
        Above is stuff for first drift drift_generations
        """
        wt_trajectory = get_trajectory(base_haplotype, drift_generations, history, pop_size)
        mut_frac = [(1-wt_traj) for wt_traj in wt_trajectory]

        tot_muts = []
        for i in range(len(mut_frac)):
            tot_muts.append(mut_frac[i]*pop_size)
        
        total_muts[repeat,:] = tot_muts

        simulate(pop, history, drug_generations, mutation_rate_drug, pop_size, seq_length, fitness, fitness_drug, sens_fit, sens_fit_drug, alphabet, base_haplotype, fitness_cost, drug=1)
        


        haplotypes_at_end_drug = list(pop.keys())
        histy_at_end_drug = []
        histy_fc_at_end_drug = []
        for haplotype in haplotypes_at_end_drug:
            if haplotype != base_haplotype:
                for i in range(pop[haplotype]):
                    histy_at_end_drug.append(sens_fit[haplotype])
                    histy_fc_at_end_drug.append(fitness[haplotype])
                
        histy_at_end_drug_repeats += histy_at_end_drug
        histy_fc_at_end_drug_repeats += histy_fc_at_end_drug
        
    
    
    end = time.time()
    print(end-start)
      
    Y_histy_at_end_drug, X_histy_at_end_drug = np.histogram(histy_at_end_drug, 40, density=True)
    C_histy_at_end_drug = [cm(((x-0)/x_span)) for x in X_histy_at_end_drug] 
    
    Y_histy_fc_at_end_drug, X_histy_fc_at_end_drug = np.histogram(histy_fc_at_end_drug, 40, density=True)
    C_histy_fc_at_end_drug = [cm(((x-0)/x_span)) for x in X_histy_fc_at_end_drug] 
    
      
    wt_trajectory = get_trajectory(base_haplotype, drift_generations+drug_generations, history, pop_size)
    mut_frac = [(1-wt_traj) for wt_traj in wt_trajectory]
    
    tot_muts = []
    for i in range(len(mut_frac)):
        tot_muts.append(mut_frac[i]*pop_size)

    print(np.mean(np.mean(total_muts)))

    print("fmax is: ")
    print(sum(fmax) / len(fmax))
        
    plt.figure(num=None, figsize=(14, 14), dpi=80, facecolor='w', edgecolor='k')
    
    plt.subplot2grid((3,2), (0,0), colspan=2)
    stacked_trajectory_plot(history, drift_generations+drug_generations, pop_size, sens_fit)
    plt.yscale("log")
    plt.ylim([5/pop_size, 1])
    
    plt.subplot2grid((3,2), (1,0), colspan=2)
    stacked_trajectory_plot_only_mutants(history, drift_generations+drug_generations, pop_size, sens_fit, base_haplotype)
    
    plt.subplot2grid((3,2), (2,0), colspan=2)
    plt.plot(range(drift_generations),tot_muts[0:drift_generations],color='red',label='total mutants')
    #plt.yscale("log")
    plt.xlabel("drift_generations")
    plt.ylabel("Total mutants")
    
       
    Y_histy_repeats, X_histy_repeats = np.histogram(histy_repeats, 40, density=True)
    C_histy_repeats = [cm(((x-0)/x_span)) for x in X_histy_repeats]
    
    Y_histy_popped_up_repeats, X_histy_popped_up_repeats = np.histogram(histy_popped_up_repeats, 40, density=True)
    C_histy_popped_up_repeats = [cm(((x-0)/x_span)) for x in X_histy_popped_up_repeats]
    
    Y_histy_at_end_drug_repeats, X_histy_at_end_drug_repeats = np.histogram(histy_at_end_drug_repeats, 40, density=True)
    C_histy_at_end_drug_repeats = [cm(((x-0)/x_span)) for x in X_histy_at_end_drug_repeats]
    
    Y_histy_fc_repeats, X_histy_fc_repeats = np.histogram(histy_fc_repeats, 40, density=True)
    C_histy_fc_repeats = [cm(((x-0)/x_span)) for x in X_histy_fc_repeats]
    
    Y_histy_fc_popped_up_repeats, X_histy_fc_popped_up_repeats = np.histogram(histy_fc_popped_up_repeats, 40, density=True)
    C_histy_fc_popped_up_repeats = [cm(((x-0)/x_span)) for x in X_histy_fc_popped_up_repeats]
    
    Y_histy_fc_at_end_drug_repeats, X_histy_fc_at_end_drug_repeats = np.histogram(histy_fc_at_end_drug_repeats, 40, density=True)
    C_histy_fc_at_end_drug_repeats = [cm(((x-0)/x_span)) for x in X_histy_fc_at_end_drug_repeats]

    df_mutfreq = pd.DataFrame()

    df_mutfreq['ecological fitness'] = X_histy_repeats[:-1]
    df_mutfreq['frequency'] = Y_histy_repeats
    df_mutfreq.to_csv('mut_freq.csv', sep=',')
    
    plt.figure(num=None, figsize=(14, 14), dpi=80, facecolor='w', edgecolor='k')
    
    plt.subplot(2,3,1)
    plt.bar(X_histy_popped_up_repeats[:-1],Y_histy_popped_up_repeats,color=C_histy_popped_up_repeats, width=X_histy_popped_up_repeats[1]-X_histy_popped_up_repeats[0])
    plt.xlim([-.05, 1.25])
    plt.ylabel('frequency')
    plt.title("Mutants that emerged during drift, all repeats")
    
    plt.subplot(2,3,2)
    plt.bar(X_histy_repeats[:-1], Y_histy_repeats, color=C_histy_repeats, width=X_histy_repeats[1]-X_histy_repeats[0])
    plt.yscale("log")
    plt.xlim([-.05, 1.25])
    plt.ylabel('frequency')
    plt.title("Mutants at the end of drift, all repeats")
    
    plt.subplot(2,3,3)
    plt.bar(X_histy_at_end_drug_repeats[:-1],Y_histy_at_end_drug_repeats,color=C_histy_at_end_drug_repeats, width=X_histy_at_end_drug_repeats[1]-X_histy_at_end_drug_repeats[0])
    plt.yscale("log")
    plt.xlim([-.05, 1.25])
    plt.ylabel('frequency')
    plt.title("Mutants at the end of drug, all repeats")
    
    
    plt.subplot(2,3,4)
    plt.bar(X_histy_fc_popped_up_repeats[:-1],Y_histy_fc_popped_up_repeats,color='black', width=X_histy_fc_popped_up_repeats[1]-X_histy_fc_popped_up_repeats[0])
    plt.xlim([-.05, 1.25])
    plt.ylabel('frequency')
    plt.title("Mutants that emerged during drift, all repeats")
    
    plt.subplot(2,3,5)
    plt.bar(X_histy_fc_repeats[:-1], Y_histy_fc_repeats, color='black', width=X_histy_fc_repeats[1]-X_histy_fc_repeats[0])
    plt.xlim([-.05, 1.25])
    plt.ylabel('frequency')
    plt.title("Mutants at the end of drift, all repeats")
    
    plt.subplot(2,3,6)
    plt.bar(X_histy_fc_at_end_drug_repeats[:-1],Y_histy_fc_at_end_drug_repeats,color='black', width=X_histy_fc_at_end_drug_repeats[1]-X_histy_fc_at_end_drug_repeats[0])
    plt.yscale("log")
    plt.xlim([-.05, 1.25])
    plt.ylabel('frequency')
    plt.title("Mutants at the end of drug, all repeats")
    
    
    #sns.jointplot(x=histy_repeats, y=histy_fc_repeats, kind='kde', space=0, fill=True, thresh=0, cmap='Blues')
    #plt.xlabel('Game growth')
    #plt.ylabel('Fitness cost')

    
    
    
    Y_test, X_test = np.histogram(histy_repeats, 48)
    C_test = [cm(((x-0)/x_span)) for x in X_test]

    print("All: ")
    print(X_test)
    print(Y_test)


    #Y_test_g1, X_test_g1 = np.histogram([i for i in histy_repeats if i > 1], 3)
    #Y_test_l1, X_test_l1 = np.histogram([i for i in histy_repeats if i <= 1], 40)

    #print("Greater than 1: ")
    #print(X_test_g1)
    #print(Y_test_g1)

    #print("Less than 1: ")
    #print(X_test_l1)
    #print(Y_test_l1)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.bar(X_test[:-1], Y_test, color=C_test, width=X_test[1]-X_test[0])
    plt.yscale("log")
    plt.subplot(1,2,2)
    plt.plot(X_test[:-1], Y_test, color='black')
    plt.yscale("log")
    
    plt.show()
    





#################################################################################################################################
"""
Simulate Wright-Fisher evolution for some amount of generations
"""
def simulate(pop, history, drift_generations, mutation_rate, pop_size, seq_length, fitness, fitness_drug, sens_fit, sens_fit_drug, alphabet, base_haplotype, fitness_cost, drug):
    clone_pop = dict(pop)
    history.append(clone_pop)
    for i in range(drift_generations):
        #print(i)
        time_step(pop, mutation_rate, pop_size, seq_length, fitness, fitness_drug, sens_fit, sens_fit_drug, alphabet, base_haplotype, fitness_cost, drug)
        clone_pop = dict(pop)
        history.append(clone_pop)
        


"""
Execute one generation of the Wright-Fisher
"""
def time_step(pop, mutation_rate, pop_size, seq_length, fitness, fitness_drug, sens_fit, sens_fit_drug, alphabet, base_haplotype, fitness_cost, drug):
    mutation_step(pop, fitness, fitness_drug, sens_fit, sens_fit_drug, mutation_rate, pop_size, seq_length, alphabet, fitness_cost)
    offspring_step(pop, pop_size, fitness, fitness_drug, sens_fit, sens_fit_drug, base_haplotype, drug)
    

#################################################################################################################################
"""
Below is the code responsible for mutation in the Wright-Fisher model, in order of function calls.
"""

"""
First step of mutation -- get a count of the mutations that occur.
"""
def mutation_step(pop, fitness, fitness_drug, sens_fit, sens_fit_drug, mutation_rate, pop_size, seq_length, alphabet, fitness_cost):
    mutation_count = get_mutation_count(mutation_rate, pop_size, seq_length)
    for i in range(mutation_count):
        mutation_event(pop, fitness, fitness_drug, sens_fit, sens_fit_drug, pop_size, seq_length, alphabet, fitness_cost)
        


"""
Draw mutation count from a poisson distribution with mean equal to average number of expected mutations.
"""
def get_mutation_count(mutation_rate, pop_size, seq_length):
    mean = mutation_rate * pop_size * seq_length
    return np.random.poisson(mean)


"""
Function that find a random haplotype to mutate and adds that new mutant to the population. Reduces mutated population by 1.
"""

def mutation_event(pop, fitness, fitness_drug, sens_fit, sens_fit_drug, pop_size, seq_length, alphabet, fitness_cost):
    haplotype = get_random_haplotype(pop, pop_size)
    if pop[haplotype] > 1:
        pop[haplotype] -= 1
        new_haplotype = get_mutant(haplotype, seq_length, alphabet)
        if new_haplotype in pop:
            pop[new_haplotype] += 1
        else:
            pop[new_haplotype] = 1
        if new_haplotype not in fitness:
            mutfc = random.random()*0.85
            fitness[new_haplotype] = copy.deepcopy(mutfc)
            sens_fit[new_haplotype] = random.random()
            #sens_fit[new_haplotype] = copy.deepcopy(mutfc)
            fitness_drug[new_haplotype] = copy.deepcopy(mutfc)
            sens_fit_drug[new_haplotype] = copy.deepcopy(mutfc)


"""
Function that find a random haplotype to mutate and adds that new mutant to the population. Reduces mutated population by 1.
"""
"""
def mutation_event(pop, fitness, fitness_drug, sens_fit, sens_fit_drug, pop_size, seq_length, alphabet, fitness_cost):
    haplotype = get_random_haplotype(pop, pop_size)
    if pop[haplotype] > 1:
        pop[haplotype] -= 1
        new_haplotype = get_mutant(haplotype, seq_length, alphabet)
        if new_haplotype in pop:
            pop[new_haplotype] += 1
        else:
            pop[new_haplotype] = 1
        if new_haplotype not in fitness:
            #mutfc = random.random()
            mutfc = 0.75
            fitness[new_haplotype] = copy.deepcopy(mutfc)
            #sens_fit[new_haplotype] = random.random()
            fitness_drug[new_haplotype] = copy.deepcopy(mutfc)
            sens_fit_drug[new_haplotype] = copy.deepcopy(mutfc)
            
            sens_fit[new_haplotype] = -1
            
            while sens_fit[new_haplotype] < 0: 
                mutfit = np.random.normal(0.5, 0.20)
                if mutfit <= 1 and mutfit >= 0:
                    sens_fit[new_haplotype] = copy.deepcopy(mutfit)
"""

"""
Chooses a random haplotype in the population that will be returned.
"""
def get_random_haplotype(pop, pop_size):
    haplotypes = list(pop.keys())
    frequencies = [x/pop_size for x in pop.values()]
    total = sum(frequencies)
    frequencies = [x / total for x in frequencies]
    return fast_choice(haplotypes, frequencies)

"""
Receives the haplotype to be mutated and returns a new haplotype with a mutation with all neighbor mutations equally probable.
"""
def get_mutant(haplotype, seq_length, alphabet):
    site = int(random.random()*seq_length)
    possible_mutations = list(alphabet)
    possible_mutations.remove(haplotype[site])
    mutation = random.choice(possible_mutations)
    new_haplotype = haplotype[:site] + mutation + haplotype[site+1:]
    return new_haplotype


#################################################################################################################################
"""
Below is the code responsible for offspring in the Wright-Fisher model, in order of function calls.
"""


"""
Gets the number of counts after an offspring step and stores them in the haplotype. If a population is reduced to zero then delete it.
"""
def offspring_step(pop, pop_size, fitness, fitness_drug, sens_fit, sens_fit_drug, base_haplotype, drug):
    haplotypes = list(pop.keys())
    counts = get_offspring_counts(pop, pop_size, fitness, fitness_drug, sens_fit, sens_fit_drug, base_haplotype, drug)
    for (haplotype, count) in zip(haplotypes, counts):
        if (count > 0):
            pop[haplotype] = count
        else:
            del pop[haplotype]

"""
Returns the new population count for each haplotype given offspring counts weighted by fitness of haplotype
"""
def get_offspring_counts(pop, pop_size, fitness, fitness_drug, sens_fit, sens_fit_drug, base_haplotype, drug):
    haplotypes = list(pop.keys())
    frequencies = [pop[haplotype]/pop_size for haplotype in haplotypes]
    
    #fitnesses = [fitness[haplotype] for haplotype in haplotypes]
    fitnesses = []
    
    
    if drug == 0:
        if base_haplotype in pop:
            for haplotype in haplotypes:
                fitnesses.append(fitness[haplotype] + (sens_fit[haplotype] - fitness[haplotype])*(pop[base_haplotype]/pop_size)) #(pop[base_haplotype]/(pop[base_haplotype]+pop[haplotype])))
        else:
            for haplotype in haplotypes:
                fitnesses.append(fitness[haplotype] + (sens_fit[haplotype] - fitness[haplotype])*(0/(0+pop[haplotype])))
                
    elif drug == 1:
        if base_haplotype in pop:
            for haplotype in haplotypes:
                fitnesses.append(fitness_drug[haplotype] + (sens_fit_drug[haplotype] - fitness_drug[haplotype])*(pop[base_haplotype]/(pop[base_haplotype]+pop[haplotype])))
        else:
            for haplotype in haplotypes:
                fitnesses.append(fitness_drug[haplotype] + (sens_fit_drug[haplotype] - fitness_drug[haplotype])*(0/(0+pop[haplotype])))
        
    weights = [x * y for x,y in zip(frequencies, fitnesses)]
    total = sum(weights)
    weights = [x / total for x in weights]
    return list(np.random.multinomial(pop_size, weights))



"""
This is faster than numpy because numpy is dog water. Self-written code to choose an item from a list with some prob.
"""
def fast_choice(options, probs):
    x = random.random()
    cum = 0
    for i, p in enumerate(probs):
        cum += p
        if x < cum:
            return options[i]
    return options[-1]




"""
Plotting stuff
"""


def get_distance(seq_a, seq_b):
    diffs = 0
    length = len(seq_a)
    assert len(seq_a) == len(seq_b)
    for chr_a, chr_b in zip(seq_a, seq_b):
        if chr_a != chr_b:
            diffs += 1
    return diffs / float(length)

def get_diversity(population, pop_size):
    haplotypes = list(population.keys())
    haplotype_count = len(haplotypes)
    diversity = 0
    for i in range(haplotype_count):
        for j in range(haplotype_count):
            haplotype_a = haplotypes[i]
            haplotype_b = haplotypes[j]
            frequency_a = population[haplotype_a] / float(pop_size)
            frequency_b = population[haplotype_b] / float(pop_size)
            frequency_pair = frequency_a * frequency_b
            diversity += frequency_pair * get_distance(haplotype_a, haplotype_b)
    return diversity

def get_diversity_trajectory(history, pop_size):
    trajectory = [get_diversity(generation, pop_size) for generation in history]
    return trajectory

def diversity_plot(history, pop_size):
    mpl.rcParams['font.size']=14
    trajectory = get_diversity_trajectory(history, pop_size)
    plt.plot(trajectory, "#447CCD")
    plt.ylabel("diversity")
    plt.xlabel("generation")

def get_divergence(population, base_haplotype, pop_size):
    haplotypes = population.keys()
    divergence = 0
    for haplotype in haplotypes:
        frequency = population[haplotype] / float(pop_size)
        divergence += frequency * get_distance(base_haplotype, haplotype)
    return divergence

def get_divergence_trajectory(history, base_haplotype, pop_size):
    trajectory = [get_divergence(generation, base_haplotype, pop_size) for generation in history]
    return trajectory

def divergence_plot(history, base_haplotype, pop_size):
    mpl.rcParams['font.size']=14
    trajectory = get_divergence_trajectory(history, base_haplotype, pop_size)
    plt.plot(trajectory, "#447CCD")
    plt.ylabel("divergence")
    plt.xlabel("generation")

def get_frequency(haplotype, generation, history, pop_size):
    pop_at_generation = history[generation]
    if haplotype in pop_at_generation:
        return pop_at_generation[haplotype]/float(pop_size)
    else:
        return 0

def get_trajectory(haplotype, drift_generations, history, pop_size):
    trajectory = [get_frequency(haplotype, gen, history, pop_size) for gen in range(drift_generations)]
    return trajectory


def get_all_haplotypes(history):
    haplotypes = set()
    for generation in history:
        for haplotype in generation:
            haplotypes.add(haplotype)
    return haplotypes

def get_game_fitness(haplotype, sens_fit):
    game_fitness = sens_fit[haplotype]
    return game_fitness
    
def get_color(game_fitness):
    if game_fitness > 1.0:
        r = 0
        g = 1
        b = 0
    elif game_fitness > 0.5:
        r = 1
        g = 1 - ((game_fitness - 0.5) / 0.5)
        b = 1 - ((game_fitness - 0.5) / 0.5)
    elif game_fitness == 0.5:
        r = 1
        g = 1
        b = 1
    elif game_fitness < 0.5:
        r = game_fitness/0.5
        g = game_fitness/0.5
        b = 1
        
    return r, g, b

def stacked_trajectory_plot(history, generations, pop_size, sens_fit, xlabel="generation"):
    

    haplotypes = get_all_haplotypes(history)
    trajectories = [get_trajectory(haplotype, generations, history, pop_size) for haplotype in haplotypes]
    game_fitnesses = [get_game_fitness(haplotype, sens_fit) for haplotype in haplotypes]
    
    game_colors = []
    for i in range(len(game_fitnesses)):
        if game_fitnesses[i] == 1 or game_fitnesses[i] == 0.5:
            game_colors.append((0,0,0))
        else:
            game_colors.append((get_color(game_fitnesses[i])))    
        
    #Sort by fitness so colors look nice on plots
    temp_traj = [x for _,x in sorted(zip(game_fitnesses,trajectories))]
    temp_games = [x for _,x in sorted(zip(game_fitnesses,game_colors))]
    
    plt.stackplot(range(generations), temp_traj, colors=temp_games)
    plt.ylim(0, 1.05)
    plt.ylabel("frequency of entire population")
    plt.xlabel(xlabel)
    
def stacked_trajectory_plot_only_mutants(history, generations, pop_size, sens_fit, base_haplotype, xlabel="generation"):
    

    haplotypes = get_all_haplotypes(history)
    trajectories = [get_trajectory(haplotype, generations, history, pop_size) for haplotype in haplotypes if haplotype != base_haplotype]
    game_fitnesses = [get_game_fitness(haplotype, sens_fit) for haplotype in haplotypes if haplotype != base_haplotype]
    
    wt_trajectory = get_trajectory(base_haplotype, generations, history, pop_size)
    dividing_value = [1 - wt_traj for wt_traj in wt_trajectory]   
    
    for i in range(len(trajectories)):
        for j in range(len(trajectories[i])):
            if j > 0:
                trajectories[i][j] /= dividing_value[j]
                
    game_colors = []
    for i in range(len(game_fitnesses)):
        game_colors.append((get_color(game_fitnesses[i])))
       
    #Sort by fitness so colors look nice on plots 
    temp_traj = [x for _, x in sorted(zip(game_fitnesses,trajectories))]
    temp_games = [x for _, x in sorted(zip(game_fitnesses,game_colors))]
    
    #plt.stackplot(range(drift_generations), trajectories, colors=game_colors)
    plt.stackplot(range(generations), temp_traj, colors=temp_games)
    plt.ylim(0, 1.05)
    plt.ylabel("frequency of only mutants")
    plt.xlabel(xlabel)    
    

# This function is not quite working, kinks to work out if I want to plot it, but stand in does okay.
def stacked_trajectory_plot_generated_mutants(history, drift_generations, pop_size, sens_fit, base_haplotype, xlabel="generation"):
    
    #haplotypes = get_all_haplotypes(history)
    haplotypes = list(sens_fit.keys())
    trajectories = [get_trajectory(haplotype, drift_generations, history, pop_size) for haplotype in haplotypes if haplotype != base_haplotype]
    game_fitnesses = [get_game_fitness(haplotype, sens_fit) for haplotype in haplotypes if haplotype != base_haplotype]   
    
    unq_mutants_count = [0]
    for i in range(1,drift_generations):
        haplo_temp = get_all_haplotypes(history[0:i+1])
        unq_mutants_count.append(len(haplo_temp)-1)
    
    
    for i in range(len(trajectories)):
        for j in range(1, len(trajectories[i])):
            if trajectories[i][j] != 0:
                for z in range(j, len(trajectories[i])):
                    trajectories[i][z] = copy.deepcopy(1/unq_mutants_count[z])
                break
                   
    game_colors = []
    for i in range(len(game_fitnesses)):
        game_colors.append((get_color(game_fitnesses[i])))
       
    #Sort by fitness so colors look nice on plots 
    temp_traj = [x for _,x in sorted(zip(game_fitnesses,trajectories))]
    temp_games = [x for _,x in sorted(zip(game_fitnesses,game_colors))]
    
    #plt.stackplot(range(drift_generations), trajectories, colors=game_colors)
    plt.stackplot(range(drift_generations), temp_traj, colors=temp_games)
    plt.ylim(0, 1.05)
    plt.ylabel("frequency of only mutants")
    plt.xlabel(xlabel)
    

def get_snp_frequency(site, generation, history, pop_size):
    minor_allele_frequency = 0.0
    pop_at_generation = history[generation]
    for haplotype in pop_at_generation.keys():
        allele = haplotype[site]
        frequency = pop_at_generation[haplotype] / float(pop_size)
        if allele != "0":
            minor_allele_frequency += frequency
    return minor_allele_frequency

def get_snp_trajectory(site, drift_generations, history, pop_size):
    trajectory = [get_snp_frequency(site, gen, history, pop_size) for gen in range(drift_generations)]
    return trajectory

def get_all_snps(history, seq_length):
    snps = set()
    for generation in history:
        for haplotype in generation:
            for site in range(seq_length):
                if haplotype[site] != "0":
                    snps.add(site)
    return snps

def snp_trajectory_plot(history, seq_length, drift_generations, pop_size, xlabel="generation"):
    colors = ["#781C86", "#571EA2", "#462EB9", "#3F47C9", "#3F63CF", "#447CCD", "#4C90C0", "#56A0AE", "#63AC9A", "#72B485", "#83BA70", "#96BD60", "#AABD52", "#BDBB48", "#CEB541", "#DCAB3C", "#E49938", "#E68133", "#E4632E", "#DF4327", "#DB2122"]
    mpl.rcParams['font.size']=18
    snps = get_all_snps(history, seq_length)
    trajectories = [get_snp_trajectory(snp, drift_generations, history, pop_size) for snp in snps]
    data = []
    for trajectory, color in zip(trajectories, itertools.cycle(colors)):
        data.append(range(drift_generations))
        data.append(trajectory)
        data.append(color)
    fig = plt.plot(*data)
    plt.ylim(0, 1)
    plt.ylabel("frequency")
    plt.xlabel(xlabel)




if __name__ == '__main__':
    main()
