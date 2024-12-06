'''This code performs a migration between two FFPopSim subpopulations, with specified migration rate, using the FFPopSim library
It needs to be ran each time the two population evolves by 1 step on their own by other mean such as recombination, mutation etc..
It can be used for both the 'lowd' and the 'highd' classes'''

import FFPopSim
import numpy as np


def remove(L, M, num):
    """
    Removes migrants from the population and updates the genotype and migrant counts.

    Parameters:
    - L: Array of clone sizes for each genotype.
    - M: Array of migrant counts for each genotype.
    - num: Number of migrants to remove.

    Returns:
    - L: Updated array of clone sizes after migration.
    - M: Updated array of migrant counts after migration.
    """
    
    # Identify indices of genotypes with clones remaining
    valid_indices = np.where(L > 0)[0]

    # If no migrants to remove or no valid genotypes with clones, return immediately
    if num <= 0 or len(valid_indices) == 0:
        return L, M

    # Perform migration by iterating over the number of migrants to be removed
    while num > 0 and len(valid_indices) > 0:
        # Calculate the probabilities of selection based on clone counts
        valid_L = L[valid_indices]
        valid_probs = valid_L / np.sum(valid_L)  # Probabilities based on clone counts

        # Select a genotype for migration
        selected_index = np.random.choice(valid_indices, size=1, p=valid_probs, replace=False)

        # Update the migrant count and reduce the clone count for the selected genotype
        M[selected_index] += 1
        L[selected_index] -= 1

        # If the selected genotype runs out of clones, remove it from valid indices
        if L[selected_index] == 0:
            valid_indices = valid_indices[valid_indices != selected_index]

        # Decrease the number of migrants to remove
        num -= 1

    return L, M


############### LOWD ###############
# These functions work with low-dimensional (LOWD) representations of genomes.
# Genotypes are represented using binary encoding.

def remove_mig_low(L, pop, num_mig):
    """
    Removes individuals from a population and returns the genotypes of the removed migrants.

    Parameters:
    - L: Number of loci in the genome.
    - pop: Population object.
    - num_mig: Number of individuals to remove.

    Returns:
    - Array of migrant genotype counts.
    """
    N = pop.N  
    if num_mig <= 0 or N == 0:  # If no migrants are to be removed, return an empty array
        return np.array([])
    
    genotypes = (pop.get_genotype_frequencies() * N).astype(int)

    # Initialize the counts for the migrants
    mig_genotypes = np.zeros(len(genotypes), dtype=int)

    genotypes, mig_genotypes =remove(genotypes,mig_genotypes,num_mig)

    # Check for any negative clone counts (this should not happen)
    if any(genotypes < 0):
        raise ValueError("Genomes have been removed while not present in the original population")

    # Remove genotypes with zero clone counts and update the population

    pop.set_genotypes(list(range(2**L)),genotypes.tolist())

    # Return the genotypes and counts of the migrants that were removed

    return mig_genotypes  # Return the genotypes of the migrants

def add_mig_low(L, pop_target, num_mig, genotype_mig):
    """
    Adds migrant individuals to the target population.

    Parameters:
    - L: Number of loci in the genome.
    - pop_target: Target population object.
    - num_mig: Number of migrants to add.
    - genotype_mig: Array of migrant genotype counts.

    Updates the target population's genotype frequencies.
    """
    N = pop_target.N  # Current population size of the target population


    # Count the genotypes in the target population
    genotypes_target = np.array(pop_target.get_genotype_frequencies()) * N
    genotypes_target += np.array(genotype_mig)  # Add migrant genotype counts to the target population

    # Update the target population with the new genotype frequencies
    pop_target.set_genotypes(list(range(2**L)), genotypes_target.tolist())

def bidirectional_migration_low(pop_A, pop_B, rate_A_to_B, rate_B_to_A):
    """
    Performs bidirectional migration between two populations.

    Parameters:
    - pop_A: Population A.
    - pop_B: Population B.
    - rate_A_to_B: Fraction of Population A migrating to Population B.
    - rate_B_to_A: Fraction of Population B migrating to Population A.
    """
    # Ensure that both populations have the same number of loci (genetic markers)
    if pop_A.number_of_loci != pop_B.number_of_loci:
        raise ValueError("Populations have different numbers of loci.")
    else:
        L = pop_A.number_of_loci  # Number of loci

    # Calculate the number of migrants from A to B and from B to A
    num_mig_A_to_B = int(rate_A_to_B * pop_A.N)  # Number of migrants from A to B
    num_mig_B_to_A = int(rate_B_to_A * pop_B.N)  # Number of migrants from B to A

    # Remove migrants from each population
    genotype_mig_A_to_B = remove_mig_low(L, pop_A, num_mig_A_to_B)
    genotype_mig_B_to_A = remove_mig_low(L, pop_B, num_mig_B_to_A)

    # Add the removed migrants to the opposite population
    add_mig_low(L, pop_B, num_mig_A_to_B, genotype_mig_A_to_B)
    add_mig_low(L, pop_A, num_mig_B_to_A, genotype_mig_B_to_A)


############### HIGHD ###############
# These functions work with high-dimensional (HIGHD) genome representations.
# Genotypes are explicitly stored as arrays, allowing for detailed manipulation.

def remove_mig_high(pop, num_mig):
    """
    Removes individuals from a population and returns the genotypes and counts of the migrants.

    Parameters:
    - pop: Population object.
    - num_mig: Number of individuals to remove.

    Returns:
    - mig_genotypes: Array of migrant genotypes.
    - mig_counts: Array of counts for each migrant genotype.
    """
    # Retrieve the genotypes and clone sizes from the population
    genotypes = pop.get_genotypes()
    clone_counts = pop.get_clone_sizes()

    # Calculate the total population size
    N = np.sum(clone_counts)

    # If no migrants are to be removed or the population is empty, return empty arrays
    if num_mig <= 0 or N == 0:
        return np.array([]), np.array([])

    # Initialize the counts for the migrants
    mig_counts = np.zeros(len(genotypes), dtype=int)

    clone_counts, mig_counts=remove(clone_counts,mig_counts,num_mig)

    # Check for any negative clone counts (this should not happen)
    if any(clone_counts < 0):
        raise ValueError("Genomes have been removed while not present in the original population")

    # Remove genotypes with zero clone counts and update the population
    remaining_counts = clone_counts[clone_counts > 0]
    remaining_genotypes = genotypes[clone_counts > 0]
    pop.set_genotypes(remaining_genotypes.tolist(), remaining_counts.tolist())

    # Return the genotypes and counts of the migrants that were removed
    mig_genotypes = genotypes[mig_counts > 0]
    mig_counts = mig_counts[mig_counts > 0]

    return mig_genotypes, mig_counts


def add_mig_high(pop_target, mig_genotypes, mig_counts):
    """
    Adds migrant individuals to the target population in an optimized manner.

    Parameters:
    - pop_target: Target population object.
    - mig_genotypes: Array of migrant genotypes.
    - mig_counts: Array of counts for each migrant genotype.
    """
    # Combine the genotypes and clone sizes of the target population with the migrant data
    genotypes_target = np.concatenate((pop_target.get_genotypes(), mig_genotypes))
    clone_sizes_target = np.concatenate((pop_target.get_clone_sizes(), mig_counts))

    # Count the unique genotypes and their respective clone sizes in the combined data
    unique_genotypes, unique_indices = np.unique(genotypes_target, axis=0, return_inverse=True)
    counts = np.bincount(unique_indices, weights=clone_sizes_target)

    # Update the target population with the new unique genotypes and their counts
    pop_target.set_genotypes(unique_genotypes.tolist(), counts.astype(int).tolist())

def bidirectional_migration_high(pop_A, pop_B, rate_A_to_B, rate_B_to_A):
    """
    Performs bidirectional migration between two populations.

    Parameters:
    - pop_A: Population A.
    - pop_B: Population B.
    - rate_A_to_B: Fraction of Population A migrating to Population B.
    - rate_B_to_A: Fraction of Population B migrating to Population A.
    """
    # Ensure that both populations have the same number of loci (genetic markers)
    if pop_A.number_of_loci != pop_B.number_of_loci:
        raise ValueError("Populations have different numbers of loci.")
    
    # Calculate the number of migrants to move from A to B and from B to A
    num_mig_A_to_B = int(rate_A_to_B * pop_A.N)
    num_mig_B_to_A = int(rate_B_to_A * pop_B.N)

    # Remove migrants from each population
    mig_genotypes_A_to_B, mig_counts_A_to_B = remove_mig_high(pop_A, num_mig_A_to_B)
    mig_genotypes_B_to_A, mig_counts_B_to_A = remove_mig_high(pop_B, num_mig_B_to_A)

    # Add the removed migrants to the opposite population
    add_mig_high(pop_B, mig_genotypes_A_to_B, mig_counts_A_to_B)
    add_mig_high(pop_A, mig_genotypes_B_to_A, mig_counts_B_to_A)