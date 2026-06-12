# -*- coding: utf-8 -*-

import cv2
import random
import numpy as np
import itertools
import copy
from const import CENTER, NUM_CORRECT_PANELS, SEED_VALUE, OPERATOR_VALUES, INTEGER_WIDTH, DATASET2 
from Calculator_Tree import (build_calculator_tree, number_sampler, fill_expression, middle_to_after)
from PIL import ImageDraw, ImageFont

np.random.seed(SEED_VALUE)  # Set the global random seed for random number generation
random.seed(SEED_VALUE)     # Set the global random seed for random number generation

def mutate_constant(value, std_dev=1):
    # Generate a new value using a normal distribution centered at the constant value
    new_value = int(random.gauss(value, std_dev))
    
    # Ensure the new value is within the acceptable range
    new_value = max(1, min(99, new_value))
    
    # Ensure the new value is not the original value
    while new_value == value:
        new_value = int(random.gauss(value, std_dev))
        new_value = max(1, min(99, new_value))
    
    return new_value


def math_parser(math_conditions, geom_conditions, prob_type, positions, show_center, reshuffle_operators_flag=False):
    """
    Prune the algebra attributes and arrange the placement of integers in each problem
    """
    num_pos = len(positions)
    pos_list = []
    gcondition_1, gcondition_2 = geom_conditions[0], geom_conditions[1]
    condition_1, condition_2 = math_conditions[0], math_conditions[1]
    constant = condition_1.integer
    operators = condition_1.operator  

    constants = constant.get_value()    # Prune a list of constants to be choose
    const_value_0 = constants[0]

    operator_value = operators.get_value()

    # ! NOTE: reshuffle_operators is not used in the current implementation
    reshuffled_operator_value = []  # Initialize it outside the block
    # reshuffle_operators_flag = True
    # if (reshuffle_operators_flag == True):
    #     original_operator_value = operator_value[:]
    #     reshuffled_operator_value = operator_value[:]

    #     # Check if all elements are the same or there's only one element
    #     if len(set(reshuffled_operator_value)) > 1: # To prevent infinite loop
    #         # Check if all elements are the same, if so mutate the operator
    #         while len(set(reshuffled_operator_value)) == 1:
    #             print("All elements are the same, reshuffling operators...")
    #             reshuffled_operator_value[0] = random.choice(OPERATOR_VALUES)

    #         while reshuffled_operator_value == original_operator_value or reshuffled_operator_value[0] == original_operator_value[0]:  # Shuffle until the operators (at least the first operator) are different
    #             random.shuffle(reshuffled_operator_value)
    #     else:
    #         print("Cannot reshuffle operators as all elements are the same or there's only one element.")

        # print("\nIn NumArrange.py, operator_value after reshuffle:\n %s" % reshuffled_operator_value)  # For debugging


    interpret_style = condition_2.interpret.get_value()
    analytical_mode, analytical_part = 0, 0
    mode = None

    if num_pos < 4:
        interpret_style = "holistic"
        condition_2.interpret.set_value("holistic")
        
    '''
    As a trade-off of problem difficulty:
    - if the constants are shown in the center of each panel, the constants in different panels are different
    - if the constants are hidden, the constants in different panels keep consistent
    MNR does not show constants, thus increase the difficulty of the problemq
    '''
    if show_center:
        const_value = constants
    else:
        const_value = [const_value_0] * NUM_CORRECT_PANELS           # Generate the number of constants according to the number of correct panels 
        mutated_const_value = [const_value_0] * NUM_CORRECT_PANELS   # Generate the number of constants according to the number of correct panels 

        mutate_constant_flag = True
        if mutate_constant_flag:
            for i in range(6, len(mutated_const_value)):    # 50% Mutate the constants in the last 8 panels
                if random.random() < 0.5:  # 50 percent chance
                     mutated_const_value[i] = mutate_constant(const_value_0)
        
        condition_1.integer.set_value(const_value)

    if interpret_style == "holistic":
        operator_list, reshuffled_operator_list, int_list_1, int_list_2, int_list_3, int_list_4, int_list_5, int_list_6, \
            integerDifferent_list_1, integerDifferent_list_2, integerDifferent_list_3, integerDifferent_list_4, \
            operatorDifferent_list_1, operatorDifferent_list_2, operatorDifferent_list_3, operatorDifferent_list_4, chosen_mutate_type_list = holistic_parser(num_pos, mutated_const_value, operator_value, reshuffled_operator_value)
        condition_1.operator.set_value(operator_list)
        pos_list = positions
    else:   
        """
        Analytical mode
        """
        analytical_part = condition_2.analytical.get_value()
        while (num_pos % analytical_part != 0) or (num_pos // analytical_part < 2):
            condition_2.analytical.sample()
            analytical_part = condition_2.analytical.get_value()

        operator_list, reshuffled_operator_list, int_list_1, int_list_2, int_list_3, int_list_4, int_list_5, int_list_6, \
            integerDifferent_list_1, integerDifferent_list_2, integerDifferent_list_3, integerDifferent_list_4, \
            operatorDifferent_list_1, operatorDifferent_list_2, operatorDifferent_list_3, operatorDifferent_list_4, chosen_mutate_type_list = analytical_parser(num_pos, const_value, operator_value, reshuffled_operator_value, analytical_part)
        condition_1.operator.set_value(operator_list)
        
        '''
        Analytical mode represents the perceptual grouping principles (e.g. Gestalt laws):
            - analytical mode == 1 refers the law of proximity (Objects grouped because they are close together)
            - analytical mode == 2 refers to the law of symmetry (Objects grouped because they are mirrored)
            - analytical mode 3 or 4 is based on analytical mode 1 or 2, but has a little adjustments.
        In the some special cases, small revision of number placement was made to render the grouping of numbers 
        more close to human intuition.
        '''
        mode = np.random.choice([1, 2])
        if mode == None:
            # Throw an exception
            raise ValueError("The mode is not set.")
        if prob_type == "Combination":
            if gcondition_1.type.get_value() == "triangle" and gcondition_2.grelation.get_value() == "include":
                if analytical_part == 2:
                    mode = 1
                elif analytical_part == 3:
                    mode = 2
        elif prob_type == "Composition":
            if gcondition_2.format.get_value() in ["circle", "square", "triangle"]:
                mode = 2

        if mode == 1:
            # print("Mode 1")
            if prob_type == "Partition":
                if gcondition_1.type.get_value() in ["hexagon", "circle"] and (gcondition_2.part.get_value() == 6 and analytical_part == 2):
                    analytical_mode = 3
                    pos_list_1 = [positions[0], positions[1], positions[2]]
                    pos_list_2 = [positions[5], positions[4], positions[3]]
                    pos_list = pos_list_1 + pos_list_2
                elif gcondition_1.type.get_value() in ["square", "circle"] and (gcondition_2.part.get_value() == 8 and analytical_part == 2):
                    analytical_mode = 3
                    pos_list_1 = [positions[0], positions[1], positions[2], positions[3]]
                    pos_list_2 = [positions[7], positions[6], positions[5], positions[4]]
                    pos_list = pos_list_1 + pos_list_2
                else:
                    analytical_mode = 1
                    pos_list = positions
            elif prob_type == "Combination":
                if gcondition_1.type.get_value() == "hexagon" and gcondition_2.grelation.get_value() == "overlap":
                    analytical_mode = 3
                    pos_list_1 = [positions[0], positions[1]]
                    pos_list_2 = [positions[3], positions[2]]
                    pos_list = pos_list_1 + pos_list_2
                else:
                    analytical_mode = 1
                    pos_list = positions
            elif prob_type == "Composition":
                analytical_mode = 1
                pos_list = positions

        elif mode == 2:
            # print("Mode 2")
            if analytical_part == 2:
                pos_list_1, pos_list_2 = [], []
                for i in range(num_pos):
                    if i % 2 == 0:
                        pos_list_1.append(positions[i])
                    elif i % 2 == 1:
                        pos_list_2.append(positions[i])
                if (prob_type == "Combination" and gcondition_2.grelation.get_value() == "include") or \
                        (prob_type == "Composition" and gcondition_2.format.get_value() == "cross"):
                    analytical_mode = 4
                    pos_list_3 = [pos_list_1[0], pos_list_1[2], pos_list_1[3], pos_list_1[1]]
                    pos_list_4 = [pos_list_2[0], pos_list_2[2], pos_list_2[3], pos_list_2[1]]
                    pos_list = pos_list_3 + pos_list_4
                else:
                    analytical_mode = 2
                    pos_list = pos_list_1 + pos_list_2
            elif analytical_part == 3:
                analytical_mode = 2
                pos_list_1, pos_list_2, pos_list_3 = [], [], []
                for i in range(num_pos):
                    if i % 3 == 0:
                        pos_list_1.append(positions[i])
                    elif i % 3 == 1:
                        pos_list_2.append(positions[i])
                    elif i % 3 == 2:
                        pos_list_3.append(positions[i])
                pos_list = pos_list_1 + pos_list_2 + pos_list_3
            elif analytical_part == 4:
                analytical_mode = 2
                pos_list_1, pos_list_2, pos_list_3, pos_list_4 = [], [], [], []
                for i in range(num_pos):
                    if i % 4 == 0:
                        pos_list_1.append(positions[i])
                    elif i % 4 == 1:
                        pos_list_2.append(positions[i])
                    elif i % 4 == 2:
                        pos_list_3.append(positions[i])
                    elif i % 4 == 3:
                        pos_list_4.append(positions[i])
                pos_list = pos_list_1 + pos_list_2 + pos_list_3 + pos_list_4

    if show_center:
        int_list_1.append(const_value[0])
        int_list_2.append(const_value[1])
        int_list_3.append(const_value[2])
        int_list_4.append(const_value[3])
        int_list_5.append(const_value[4])
        int_list_6.append(const_value[5])
        integerDifferent_list_1.append(const_value[4])
        integerDifferent_list_2.append(const_value[5])
        integerDifferent_list_3.append(const_value[6])
        integerDifferent_list_4.append(const_value[7])
        operatorDifferent_list_1.append(const_value[8])
        operatorDifferent_list_2.append(const_value[9])
        operatorDifferent_list_3.append(const_value[10])
        operatorDifferent_list_4.append(const_value[11])
        pos_list.append(CENTER)

    return interpret_style, mode, analytical_part, operator_list, reshuffled_operator_list, \
            int_list_1, int_list_2, int_list_3, int_list_4, int_list_5, int_list_6, \
            integerDifferent_list_1, integerDifferent_list_2, integerDifferent_list_3, integerDifferent_list_4, \
            operatorDifferent_list_1, operatorDifferent_list_2, operatorDifferent_list_3, operatorDifferent_list_4, pos_list, chosen_mutate_type_list


### HELPER FUNCTIONS ###
# generate the numbers and operational relations for holistic problems
def holistic_parser(num_pos, const_value, operator_value, reshuffled_operator_value):
    num_blanks = num_pos - 1
    operator_list = operator_prune(num_blanks, operator_value)
    reshuffled_operator_list = [] # ! THIS IS ALSO NOT NEEDED, initialiazed as empty value
    
    # START========================================================================================================================

    mutated_operator_lists = []
    # Randomly select a mutation type from mutate_type = ['MutOp', 'MutPar', 'MutBoth']
    mutate_type = ['MutOp', 'MutPar', 'MutBoth']
    chosen_mutate_type_list = np.random.choice(mutate_type, size=8, replace=True)

    # Choose randomly whether to mutate two or three operators if operator is more than 1
    # Filter out parenthesis operators
    non_parenthesis_operators_index_position = [i for i, op in enumerate(operator_list) if op not in ['(', ')']]
    
    if len(non_parenthesis_operators_index_position) > 2:
        # Randomly choose to mutate either 2 or 3 operators
        num_of_operators_to_mutate = random.choice([2, 3])
    elif len(non_parenthesis_operators_index_position) == 2:
        num_of_operators_to_mutate = 2
    else:
        num_of_operators_to_mutate = 1

    # Randomly select the indices to mutate 
    opreator_indices_to_mutate = np.random.choice(non_parenthesis_operators_index_position, num_of_operators_to_mutate, replace=False)

    mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type_list[0], opreator_indices_to_mutate))   # 1
    mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type_list[1], opreator_indices_to_mutate))   # 2
    mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type_list[2], opreator_indices_to_mutate))   # 3
    mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type_list[3], opreator_indices_to_mutate))   # 4
    mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type_list[4], opreator_indices_to_mutate))   # 5
    mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type_list[5], opreator_indices_to_mutate))   # 6
    mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type_list[6], opreator_indices_to_mutate))   # 7
    mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type_list[7], opreator_indices_to_mutate))   # 8

    # END========================================================================================================================

    integer_list_1, integer_list_2, integer_list_3, integer_list_4, integer_list_5, integer_list_6 = [], [], [], [], [], [] # Generate 6 correct integer lists
    integerDifferent_integer_list_1, integerDifferent_integer_list_2, integerDifferent_integer_list_3, integerDifferent_integer_list_4 = [], [], [], []
    operatorDifferent_integer_list_1, operatorDifferent_integer_list_2, operatorDifferent_integer_list_3, operatorDifferent_integer_list_4 = [], [], [], []
    
    qualified_1, qualified_2, qualified_3, qualified_4, qualified_5, qualified_6 = False, False, False, False, False, False
    integerDifferent_qualified_1, integerDifferent_qualified_2, integerDifferent_qualified_3, integerDifferent_qualified_4 = False, False, False, False
    operatorDifferent_qualified_1, operatorDifferent_qualified_2, operatorDifferent_qualified_3, operatorDifferent_qualified_4 = False, False, False, False
    
    while (
        (not qualified_1) or (not qualified_2) or (not qualified_3) or (not qualified_4) or (not qualified_5) or (not qualified_6) or
        (not integerDifferent_qualified_1) or (not integerDifferent_qualified_2) or (not integerDifferent_qualified_3) or (not integerDifferent_qualified_4) or 
        (not operatorDifferent_qualified_1) or (not operatorDifferent_qualified_2) or (not operatorDifferent_qualified_3) or (not operatorDifferent_qualified_4)
    ):    
        operator_list = operator_prune(num_blanks, operator_value)

        # START========================================================================================================================

        mutated_operator_lists = []
        # Randomly select a mutation type from mutate_type = ['MutOp', 'MutPar', 'MutBoth']
        mutate_type = ['MutOp', 'MutPar', 'MutBoth']
        chosen_mutate_type_list = np.random.choice(mutate_type, size=8, replace=True)

        # Choose randomly whether to mutate two or three operators if operator is more than 1
        # Filter out parenthesis operators
        non_parenthesis_operators_index_position = [i for i, op in enumerate(operator_list) if op not in ['(', ')']]
        if len(non_parenthesis_operators_index_position) > 2:
            # Randomly choose to mutate either 2 or 3 operators
            num_of_operators_to_mutate = random.choice([2, 3])
        elif len(non_parenthesis_operators_index_position) == 2:
            num_of_operators_to_mutate = 2
        else:
            num_of_operators_to_mutate = 1

        # Randomly select the indices to mutate 
        opreator_indices_to_mutate = np.random.choice(non_parenthesis_operators_index_position, num_of_operators_to_mutate, replace=False)

        mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type_list[0], opreator_indices_to_mutate))   # 1
        mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type_list[1], opreator_indices_to_mutate))   # 2
        mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type_list[2], opreator_indices_to_mutate))   # 3
        mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type_list[3], opreator_indices_to_mutate))   # 4
        mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type_list[4], opreator_indices_to_mutate))   # 5
        mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type_list[5], opreator_indices_to_mutate))   # 6
        mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type_list[6], opreator_indices_to_mutate))   # 7
        mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type_list[7], opreator_indices_to_mutate))   # 8

        # END========================================================================================================================

        integer_list_1, qualified_1 = int_generator(operator_list, const_value[0])
        integer_list_2, qualified_2 = int_generator(operator_list, const_value[1])
        integer_list_3, qualified_3 = int_generator(operator_list, const_value[2])
        integer_list_4, qualified_4 = int_generator(operator_list, const_value[3])
        integer_list_5, qualified_5 = int_generator(operator_list, const_value[4])
        integer_list_6, qualified_6 = int_generator(operator_list, const_value[5])

        integerDifferent_integer_list_1, integerDifferent_qualified_1 = int_generator(mutated_operator_lists[0], const_value[6])
        integerDifferent_integer_list_2, integerDifferent_qualified_2 = int_generator(mutated_operator_lists[1], const_value[7])
        integerDifferent_integer_list_3, integerDifferent_qualified_3 = int_generator(mutated_operator_lists[2], const_value[8])
        integerDifferent_integer_list_4, integerDifferent_qualified_4 = int_generator(mutated_operator_lists[3], const_value[9])
        operatorDifferent_integer_list_1, operatorDifferent_qualified_1 = int_generator(mutated_operator_lists[4], const_value[10])
        operatorDifferent_integer_list_2, operatorDifferent_qualified_2 = int_generator(mutated_operator_lists[5], const_value[11])
        operatorDifferent_integer_list_3, operatorDifferent_qualified_3 = int_generator(mutated_operator_lists[6], const_value[12])
        operatorDifferent_integer_list_4, operatorDifferent_qualified_4 = int_generator(mutated_operator_lists[7], const_value[13])

        random.shuffle(operator_value)              
        # random.shuffle(reshuffled_operator_value)

    return operator_list, reshuffled_operator_list, integer_list_1, integer_list_2, integer_list_3, integer_list_4, integer_list_5, integer_list_6, \
        integerDifferent_integer_list_1, integerDifferent_integer_list_2, integerDifferent_integer_list_3, integerDifferent_integer_list_4, \
        operatorDifferent_integer_list_1, operatorDifferent_integer_list_2, operatorDifferent_integer_list_3, operatorDifferent_integer_list_4, \
        chosen_mutate_type_list


# generate the numbers and operational relations for analytical problem
def analytical_parser(num_pos, const_value, operator_value, reshuffled_operator_value, analytical_part):
    num_blanks = num_pos // analytical_part - 1
    operator_list = operator_prune(num_blanks, operator_value)
    reshuffled_operator_list = [] # ! THIS IS ALSO NOT NEEDED, initialiazed as empty value
    # reshuffled_operator_list = operator_prune(num_blanks, reshuffled_operator_value)

    # START========================================================================================================================

    mutated_operator_lists = []
    mutate_type = ['MutFalse']
    chosen_mutate_type_list = np.random.choice(mutate_type, size=8, replace=True)

    # Assert that the list contains exactly 8 instances of 'MutFalse'
    assert len(chosen_mutate_type_list) == 8 and all(mt == 'MutFalse' for mt in chosen_mutate_type_list), \
        "For analytical interpretation, mutate_type must contain exactly 8 instances of 'MutFalse'"
    
    # Choose randomly whether to mutate two or three operators if operator is more than 1
    # Filter out parenthesis operators
    non_parenthesis_operators_index_position = [i for i, op in enumerate(operator_list) if op not in ['(', ')']]
    if len(non_parenthesis_operators_index_position) > 2:
        # Randomly choose to mutate either 2 or 3 operators
        num_of_operators_to_mutate = random.choice([2, 3])
    elif len(non_parenthesis_operators_index_position) == 2:
        num_of_operators_to_mutate = 2
    else:
        num_of_operators_to_mutate = 1

    # Randomly select the indices to mutate 
    opreator_indices_to_mutate = np.random.choice(non_parenthesis_operators_index_position, num_of_operators_to_mutate, replace=False)

    mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type_list[0], opreator_indices_to_mutate))   # 1
    mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type_list[1], opreator_indices_to_mutate))   # 2
    mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type_list[2], opreator_indices_to_mutate))   # 3
    mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type_list[3], opreator_indices_to_mutate))   # 4
    mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type_list[4], opreator_indices_to_mutate))   # 5
    mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type_list[5], opreator_indices_to_mutate))   # 6
    mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type_list[6], opreator_indices_to_mutate))   # 7
    mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type_list[7], opreator_indices_to_mutate))   # 8

    # END========================================================================================================================

    integer_list_1, integer_list_2, integer_list_3, integer_list_4, integer_list_5, integer_list_6 = [], [], [], [], [], []
    integerDifferent_integer_list_1, integerDifferent_integer_list_2, integerDifferent_integer_list_3, integerDifferent_integer_list_4 = [], [], [], []
    operatorDifferent_integer_list_1, operatorDifferent_integer_list_2, operatorDifferent_integer_list_3, operatorDifferent_integer_list_4 = [], [], [], []

    qualified_1, qualified_2, qualified_3, qualified_4, qualified_5, qualified_6, qualified_7, qualified_8 = False, False, False, False, False, False, False, False
    qualified_9, qualified_10, qualified_11, qualified_12, qualified_13, qualified_14, qualified_15, qualified_16 = False, False, False, False, False, False, False, False
    qualified_17, qualified_18, qualified_19, qualified_20, qualified_21, qualified_22, qualified_23, qualified_24 = False, False, False, False, False, False, False, False

    integerDifferent_qualified_1, integerDifferent_qualified_2, integerDifferent_qualified_3, integerDifferent_qualified_4 = False, False, False, False
    integerDifferent_qualified_5, integerDifferent_qualified_6, integerDifferent_qualified_7, integerDifferent_qualified_8 = False, False, False, False
    integerDifferent_qualified_9, integerDifferent_qualified_10, integerDifferent_qualified_11, integerDifferent_qualified_12 = False, False, False, False
    integerDifferent_qualified_13, integerDifferent_qualified_14, integerDifferent_qualified_15, integerDifferent_qualified_16 = False, False, False, False

    operatorDifferent_qualified_1, operatorDifferent_qualified_2, operatorDifferent_qualified_3, operatorDifferent_qualified_4 = False, False, False, False
    operatorDifferent_qualified_5, operatorDifferent_qualified_6, operatorDifferent_qualified_7, operatorDifferent_qualified_8 = False, False, False, False
    operatorDifferent_qualified_9, operatorDifferent_qualified_10, operatorDifferent_qualified_11, operatorDifferent_qualified_12 = False, False, False, False
    operatorDifferent_qualified_13, operatorDifferent_qualified_14, operatorDifferent_qualified_15, operatorDifferent_qualified_16 = False, False, False, False

    if analytical_part == 2:
        while not (qualified_1 and qualified_2 and qualified_3 and qualified_4 and qualified_5 and qualified_6 
                    and qualified_7 and qualified_8 and qualified_9 and qualified_10 and qualified_11 and qualified_12
                    and integerDifferent_qualified_1 and integerDifferent_qualified_2 and integerDifferent_qualified_3 and integerDifferent_qualified_4
                    and integerDifferent_qualified_5 and integerDifferent_qualified_6 and integerDifferent_qualified_7 and integerDifferent_qualified_8
                    and operatorDifferent_qualified_1 and operatorDifferent_qualified_2 and operatorDifferent_qualified_3 and operatorDifferent_qualified_4
                    and operatorDifferent_qualified_5 and operatorDifferent_qualified_6 and operatorDifferent_qualified_7 and operatorDifferent_qualified_8):
            operator_list = operator_prune(num_blanks, operator_value)
            # reshuffled_operator_list = operator_prune(num_blanks, reshuffled_operator_value)

            # START========================================================================================================================

            mutated_operator_lists = []
            mutate_type = ['MutFalse']
            chosen_mutate_type_list = np.random.choice(mutate_type, size=16, replace=True)

            # Assert that the list contains exactly 16 instances of 'MutFalse'
            assert len(chosen_mutate_type_list) == 16 and all(mt == 'MutFalse' for mt in chosen_mutate_type_list), \
                "For analytical interpretation, 2 part, mutate_type must contain exactly 16 instances of 'MutFalse'"
            
            # Choose randomly whether to mutate two or three operators if operator is more than 1
            # Filter out parenthesis operators
            non_parenthesis_operators_index_position = [i for i, op in enumerate(operator_list) if op not in ['(', ')']]
            if len(non_parenthesis_operators_index_position) > 2:
                # Randomly choose to mutate either 2 or 3 operators
                num_of_operators_to_mutate = random.choice([2, 3])
            elif len(non_parenthesis_operators_index_position) == 2:
                num_of_operators_to_mutate = 2
            else:
                num_of_operators_to_mutate = 1

            # Randomly select the indices to mutate 
            opreator_indices_to_mutate = np.random.choice(non_parenthesis_operators_index_position, num_of_operators_to_mutate, replace=False)

            for chosen_mutate_type in chosen_mutate_type_list:
                mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type, opreator_indices_to_mutate))
            

            # END========================================================================================================================

            part_1_1, qualified_1 = int_generator(operator_list, const_value[0])    # For Integer List 1
            part_1_2, qualified_2 = int_generator(operator_list, const_value[0])    # For Integer List 1
            part_2_1, qualified_3 = int_generator(operator_list, const_value[1])    # For Integer List 2
            part_2_2, qualified_4 = int_generator(operator_list, const_value[1])    # For Integer List 2
            part_3_1, qualified_5 = int_generator(operator_list, const_value[2])    # For Integer List 3
            part_3_2, qualified_6 = int_generator(operator_list, const_value[2])    # For Integer List 3
            part_4_1, qualified_7 = int_generator(operator_list, const_value[3])    # For Integer List 4
            part_4_2, qualified_8 = int_generator(operator_list, const_value[3])    # For Integer List 4
            part_5_1, qualified_9 = int_generator(operator_list, const_value[4])    # For Integer List 5
            part_5_2, qualified_10 = int_generator(operator_list, const_value[4])    # For Integer List 5
            part_6_1, qualified_11 = int_generator(operator_list, const_value[5])    # For Integer List 6
            part_6_2, qualified_12 = int_generator(operator_list, const_value[5])    # For Integer List 6

            integerDifferent_part_1_1, integerDifferent_qualified_1 = int_generator(mutated_operator_lists[0], const_value[6])
            integerDifferent_part_1_2, integerDifferent_qualified_2 = int_generator(mutated_operator_lists[1], const_value[6])
            integerDifferent_part_2_1, integerDifferent_qualified_3 = int_generator(mutated_operator_lists[2], const_value[7])
            integerDifferent_part_2_2, integerDifferent_qualified_4 = int_generator(mutated_operator_lists[3], const_value[7])
            integerDifferent_part_3_1, integerDifferent_qualified_5 = int_generator(mutated_operator_lists[4], const_value[8])
            integerDifferent_part_3_2, integerDifferent_qualified_6 = int_generator(mutated_operator_lists[5], const_value[8])
            integerDifferent_part_4_1, integerDifferent_qualified_7 = int_generator(mutated_operator_lists[6], const_value[9])
            integerDifferent_part_4_2, integerDifferent_qualified_8 = int_generator(mutated_operator_lists[7], const_value[9])

            operatorDifferent_part_1_1, operatorDifferent_qualified_1 = int_generator(mutated_operator_lists[8], const_value[10])
            operatorDifferent_part_1_2, operatorDifferent_qualified_2 = int_generator(mutated_operator_lists[9], const_value[10])
            operatorDifferent_part_2_1, operatorDifferent_qualified_3 = int_generator(mutated_operator_lists[10], const_value[11])
            operatorDifferent_part_2_2, operatorDifferent_qualified_4 = int_generator(mutated_operator_lists[11], const_value[11])
            operatorDifferent_part_3_1, operatorDifferent_qualified_5 = int_generator(mutated_operator_lists[12], const_value[12])
            operatorDifferent_part_3_2, operatorDifferent_qualified_6 = int_generator(mutated_operator_lists[13], const_value[12])
            operatorDifferent_part_4_1, operatorDifferent_qualified_7 = int_generator(mutated_operator_lists[14], const_value[13])
            operatorDifferent_part_4_2, operatorDifferent_qualified_8 = int_generator(mutated_operator_lists[15], const_value[13])

            random.shuffle(operator_value)
            # random.shuffle(reshuffled_operator_value)

        integer_list_1 = part_1_1 + part_1_2
        integer_list_2 = part_2_1 + part_2_2
        integer_list_3 = part_3_1 + part_3_2
        integer_list_4 = part_4_1 + part_4_2
        integer_list_5 = part_5_1 + part_5_2
        integer_list_6 = part_6_1 + part_6_2

        integerDifferent_integer_list_1 = integerDifferent_part_1_1 + integerDifferent_part_1_2
        integerDifferent_integer_list_2 = integerDifferent_part_2_1 + integerDifferent_part_2_2
        integerDifferent_integer_list_3 = integerDifferent_part_3_1 + integerDifferent_part_3_2
        integerDifferent_integer_list_4 = integerDifferent_part_4_1 + integerDifferent_part_4_2

        operatorDifferent_integer_list_1 = operatorDifferent_part_1_1 + operatorDifferent_part_1_2
        operatorDifferent_integer_list_2 = operatorDifferent_part_2_1 + operatorDifferent_part_2_2
        operatorDifferent_integer_list_3 = operatorDifferent_part_3_1 + operatorDifferent_part_3_2
        operatorDifferent_integer_list_4 = operatorDifferent_part_4_1 + operatorDifferent_part_4_2

    if analytical_part == 3:
        while not (qualified_1 and qualified_2 and qualified_3 and qualified_4 and qualified_5 and qualified_6
                    and qualified_7 and qualified_8 and qualified_9 and qualified_10 and qualified_11 and qualified_12
                    and qualified_13 and qualified_14 and qualified_15 and qualified_16 and qualified_17 and qualified_18
                    and integerDifferent_qualified_1 and integerDifferent_qualified_2 and integerDifferent_qualified_3 and integerDifferent_qualified_4
                    and integerDifferent_qualified_5 and integerDifferent_qualified_6 and integerDifferent_qualified_7 and integerDifferent_qualified_8
                    and integerDifferent_qualified_9 and integerDifferent_qualified_10 and integerDifferent_qualified_11 and integerDifferent_qualified_12
                    and operatorDifferent_qualified_1 and operatorDifferent_qualified_2 and operatorDifferent_qualified_3 and operatorDifferent_qualified_4
                    and operatorDifferent_qualified_5 and operatorDifferent_qualified_6 and operatorDifferent_qualified_7 and operatorDifferent_qualified_8
                    and operatorDifferent_qualified_9 and operatorDifferent_qualified_10 and operatorDifferent_qualified_11 and operatorDifferent_qualified_12):
            operator_list = operator_prune(num_blanks, operator_value)
            # reshuffled_operator_list = operator_prune(num_blanks, reshuffled_operator_value)

            # START========================================================================================================================

            mutated_operator_lists = []
            mutate_type = ['MutFalse']
            chosen_mutate_type_list = np.random.choice(mutate_type, size=24, replace=True)

            # Assert that the list contains exactly 24 instances of 'MutFalse'
            assert len(chosen_mutate_type_list) == 24 and all(mt == 'MutFalse' for mt in chosen_mutate_type_list), \
                "For analytical interpretation, 3 part, mutate_type must contain exactly 24 instances of 'MutFalse'"
            
            # Choose randomly whether to mutate two or three operators if operator is more than 1
            # Filter out parenthesis operators
            non_parenthesis_operators_index_position = [i for i, op in enumerate(operator_list) if op not in ['(', ')']]
            if len(non_parenthesis_operators_index_position) > 2:
                # Randomly choose to mutate either 2 or 3 operators
                num_of_operators_to_mutate = random.choice([2, 3])
            elif len(non_parenthesis_operators_index_position) == 2:
                num_of_operators_to_mutate = 2
            else:
                num_of_operators_to_mutate = 1

            # Randomly select the indices to mutate 
            opreator_indices_to_mutate = np.random.choice(non_parenthesis_operators_index_position, num_of_operators_to_mutate, replace=False)
   
            for chosen_mutate_type in chosen_mutate_type_list:
                mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type, opreator_indices_to_mutate))
            

            # END========================================================================================================================

            part_1_1, qualified_1 = int_generator(operator_list, const_value[0])     # For Integer List 1
            part_1_2, qualified_2 = int_generator(operator_list, const_value[0])     # For Integer List 1
            part_1_3, qualified_3 = int_generator(operator_list, const_value[0])     # For Integer List 1
            part_2_1, qualified_4 = int_generator(operator_list, const_value[1])     # For Integer List 2
            part_2_2, qualified_5 = int_generator(operator_list, const_value[1])     # For Integer List 2
            part_2_3, qualified_6 = int_generator(operator_list, const_value[1])     # For Integer List 2
            part_3_1, qualified_7 = int_generator(operator_list, const_value[2])     # For Integer List 3
            part_3_2, qualified_8 = int_generator(operator_list, const_value[2])     # For Integer List 3
            part_3_3, qualified_9 = int_generator(operator_list, const_value[2])     # For Integer List 3
            part_4_1, qualified_10 = int_generator(operator_list, const_value[3])    # For Integer List 4
            part_4_2, qualified_11 = int_generator(operator_list, const_value[3])    # For Integer List 4
            part_4_3, qualified_12 = int_generator(operator_list, const_value[3])    # For Integer List 4
            part_5_1, qualified_13 = int_generator(operator_list, const_value[4])    # For Integer List 5
            part_5_2, qualified_14 = int_generator(operator_list, const_value[4])    # For Integer List 5
            part_5_3, qualified_15 = int_generator(operator_list, const_value[4])    # For Integer List 5
            part_6_1, qualified_16 = int_generator(operator_list, const_value[5])    # For Integer List 6
            part_6_2, qualified_17 = int_generator(operator_list, const_value[5])    # For Integer List 6
            part_6_3, qualified_18 = int_generator(operator_list, const_value[5])    # For Integer List 6

            integerDifferent_part_1_1, integerDifferent_qualified_1 = int_generator(mutated_operator_lists[0], const_value[6])
            integerDifferent_part_1_2, integerDifferent_qualified_2 = int_generator(mutated_operator_lists[1], const_value[6])
            integerDifferent_part_1_3, integerDifferent_qualified_3 = int_generator(mutated_operator_lists[2], const_value[6])
            integerDifferent_part_2_1, integerDifferent_qualified_4 = int_generator(mutated_operator_lists[3], const_value[7])
            integerDifferent_part_2_2, integerDifferent_qualified_5 = int_generator(mutated_operator_lists[4], const_value[7])
            integerDifferent_part_2_3, integerDifferent_qualified_6 = int_generator(mutated_operator_lists[5], const_value[7])
            integerDifferent_part_3_1, integerDifferent_qualified_7 = int_generator(mutated_operator_lists[6], const_value[8])
            integerDifferent_part_3_2, integerDifferent_qualified_8 = int_generator(mutated_operator_lists[7], const_value[8])
            integerDifferent_part_3_3, integerDifferent_qualified_9 = int_generator(mutated_operator_lists[8], const_value[8])
            integerDifferent_part_4_1, integerDifferent_qualified_10 = int_generator(mutated_operator_lists[9], const_value[9])
            integerDifferent_part_4_2, integerDifferent_qualified_11 = int_generator(mutated_operator_lists[10], const_value[9])
            integerDifferent_part_4_3, integerDifferent_qualified_12 = int_generator(mutated_operator_lists[11], const_value[9])

            operatorDifferent_part_1_1, operatorDifferent_qualified_1 = int_generator(mutated_operator_lists[12], const_value[10])
            operatorDifferent_part_1_2, operatorDifferent_qualified_2 = int_generator(mutated_operator_lists[13], const_value[10])
            operatorDifferent_part_1_3, operatorDifferent_qualified_3 = int_generator(mutated_operator_lists[14], const_value[10])
            operatorDifferent_part_2_1, operatorDifferent_qualified_4 = int_generator(mutated_operator_lists[15], const_value[11])
            operatorDifferent_part_2_2, operatorDifferent_qualified_5 = int_generator(mutated_operator_lists[16], const_value[11])
            operatorDifferent_part_2_3, operatorDifferent_qualified_6 = int_generator(mutated_operator_lists[17], const_value[11])
            operatorDifferent_part_3_1, operatorDifferent_qualified_7 = int_generator(mutated_operator_lists[18], const_value[12])
            operatorDifferent_part_3_2, operatorDifferent_qualified_8 = int_generator(mutated_operator_lists[19], const_value[12])
            operatorDifferent_part_3_3, operatorDifferent_qualified_9 = int_generator(mutated_operator_lists[20], const_value[12])
            operatorDifferent_part_4_1, operatorDifferent_qualified_10 = int_generator(mutated_operator_lists[21], const_value[13])
            operatorDifferent_part_4_2, operatorDifferent_qualified_11 = int_generator(mutated_operator_lists[22], const_value[13])
            operatorDifferent_part_4_3, operatorDifferent_qualified_12 = int_generator(mutated_operator_lists[23], const_value[13])

            random.shuffle(operator_value)
            # random.shuffle(reshuffled_operator_value)
        integer_list_1 = part_1_1 + part_1_2 + part_1_3
        integer_list_2 = part_2_1 + part_2_2 + part_2_3
        integer_list_3 = part_3_1 + part_3_2 + part_3_3
        integer_list_4 = part_4_1 + part_4_2 + part_4_3
        integer_list_5 = part_5_1 + part_5_2 + part_5_3
        integer_list_6 = part_6_1 + part_6_2 + part_6_3

        integerDifferent_integer_list_1 = integerDifferent_part_1_1 + integerDifferent_part_1_2 + integerDifferent_part_1_3
        integerDifferent_integer_list_2 = integerDifferent_part_2_1 + integerDifferent_part_2_2 + integerDifferent_part_2_3
        integerDifferent_integer_list_3 = integerDifferent_part_3_1 + integerDifferent_part_3_2 + integerDifferent_part_3_3
        integerDifferent_integer_list_4 = integerDifferent_part_4_1 + integerDifferent_part_4_2 + integerDifferent_part_4_3

        operatorDifferent_integer_list_1 = operatorDifferent_part_1_1 + operatorDifferent_part_1_2 + operatorDifferent_part_1_3
        operatorDifferent_integer_list_2 = operatorDifferent_part_2_1 + operatorDifferent_part_2_2 + operatorDifferent_part_2_3
        operatorDifferent_integer_list_3 = operatorDifferent_part_3_1 + operatorDifferent_part_3_2 + operatorDifferent_part_3_3
        operatorDifferent_integer_list_4 = operatorDifferent_part_4_1 + operatorDifferent_part_4_2 + operatorDifferent_part_4_3

    if analytical_part == 4:
        while not (qualified_1 and qualified_2 and qualified_3 and qualified_4 
                   and qualified_5 and qualified_6 and qualified_7 and qualified_8 
                   and qualified_9 and qualified_10 and qualified_11 and qualified_12
                   and qualified_13 and qualified_14 and qualified_15 and qualified_16
                   and qualified_17 and qualified_18 and qualified_19 and qualified_20
                   and qualified_21 and qualified_22 and qualified_23 and qualified_24
                   and integerDifferent_qualified_1 and integerDifferent_qualified_2 and integerDifferent_qualified_3 and integerDifferent_qualified_4
                   and integerDifferent_qualified_5 and integerDifferent_qualified_6 and integerDifferent_qualified_7 and integerDifferent_qualified_8
                   and integerDifferent_qualified_9 and integerDifferent_qualified_10 and integerDifferent_qualified_11 and integerDifferent_qualified_12
                   and integerDifferent_qualified_13 and integerDifferent_qualified_14 and integerDifferent_qualified_15 and integerDifferent_qualified_16
                   and operatorDifferent_qualified_1 and operatorDifferent_qualified_2 and operatorDifferent_qualified_3 and operatorDifferent_qualified_4
                   and operatorDifferent_qualified_5 and operatorDifferent_qualified_6 and operatorDifferent_qualified_7 and operatorDifferent_qualified_8
                   and operatorDifferent_qualified_9 and operatorDifferent_qualified_10 and operatorDifferent_qualified_11 and operatorDifferent_qualified_12
                   and operatorDifferent_qualified_13 and operatorDifferent_qualified_14 and operatorDifferent_qualified_15 and operatorDifferent_qualified_16):
            operator_list = operator_prune(num_blanks, operator_value)
            # reshuffled_operator_list = operator_prune(num_blanks, reshuffled_operator_value)

            # START========================================================================================================================

            mutated_operator_lists = []
            mutate_type = ['MutFalse']  # For analytical problem, only use 'MutFalse'
            chosen_mutate_type_list = np.random.choice(mutate_type, size=32, replace=True)

            # Assert that the list contains exactly 32 instances of 'MutFalse'
            assert len(chosen_mutate_type_list) == 32 and all(mt == 'MutFalse' for mt in chosen_mutate_type_list), \
                "For analytical interpretation, 4 part, mutate_type must contain exactly 32 instances of 'MutFalse'"
            
            # Choose randomly whether to mutate two or three operators if operator is more than 1
            # Filter out parenthesis operators
            non_parenthesis_operators_index_position = [i for i, op in enumerate(operator_list) if op not in ['(', ')']]
            if len(non_parenthesis_operators_index_position) > 2:
                # Randomly choose to mutate either 2 or 3 operators
                num_of_operators_to_mutate = random.choice([2, 3])
            elif len(non_parenthesis_operators_index_position) == 2:
                num_of_operators_to_mutate = 2
            else:
                num_of_operators_to_mutate = 1

            # Randomly select the indices to mutate 
            opreator_indices_to_mutate = np.random.choice(non_parenthesis_operators_index_position, num_of_operators_to_mutate, replace=False)

            for chosen_mutate_type in chosen_mutate_type_list:
                mutated_operator_lists.append(mutated_operator_prune(num_blanks, operator_list, chosen_mutate_type, opreator_indices_to_mutate))

            # END========================================================================================================================

            part_1_1, qualified_1 = int_generator(operator_list, const_value[0])    # For Integer List 1
            part_1_2, qualified_2 = int_generator(operator_list, const_value[0])    # For Integer List 1
            part_1_3, qualified_3 = int_generator(operator_list, const_value[0])    # For Integer List 1
            part_1_4, qualified_4 = int_generator(operator_list, const_value[0])    # For Integer List 1
            part_2_1, qualified_5 = int_generator(operator_list, const_value[1])    # For Integer List 2
            part_2_2, qualified_6 = int_generator(operator_list, const_value[1])    # For Integer List 2
            part_2_3, qualified_7 = int_generator(operator_list, const_value[1])    # For Integer List 2
            part_2_4, qualified_8 = int_generator(operator_list, const_value[1])    # For Integer List 2
            part_3_1, qualified_9 = int_generator(operator_list, const_value[2])    # For Integer List 3
            part_3_2, qualified_10 = int_generator(operator_list, const_value[2])   # For Integer List 3
            part_3_3, qualified_11 = int_generator(operator_list, const_value[2])   # For Integer List 3
            part_3_4, qualified_12 = int_generator(operator_list, const_value[2])   # For Integer List 3
            part_4_1, qualified_13 = int_generator(operator_list, const_value[3])   # For Integer List 4
            part_4_2, qualified_14 = int_generator(operator_list, const_value[3])   # For Integer List 4
            part_4_3, qualified_15 = int_generator(operator_list, const_value[3])   # For Integer List 4
            part_4_4, qualified_16 = int_generator(operator_list, const_value[3])   # For Integer List 4
            part_5_1, qualified_17 = int_generator(operator_list, const_value[4])   # For Integer List 5
            part_5_2, qualified_18 = int_generator(operator_list, const_value[4])   # For Integer List 5
            part_5_3, qualified_19 = int_generator(operator_list, const_value[4])   # For Integer List 5
            part_5_4, qualified_20 = int_generator(operator_list, const_value[4])   # For Integer List 5
            part_6_1, qualified_21 = int_generator(operator_list, const_value[5])   # For Integer List 6
            part_6_2, qualified_22 = int_generator(operator_list, const_value[5])   # For Integer List 6
            part_6_3, qualified_23 = int_generator(operator_list, const_value[5])   # For Integer List 6
            part_6_4, qualified_24 = int_generator(operator_list, const_value[5])   # For Integer List 6


            integerDifferent_part_1_1, integerDifferent_qualified_1 = int_generator(mutated_operator_lists[0], const_value[6])
            integerDifferent_part_1_2, integerDifferent_qualified_2 = int_generator(mutated_operator_lists[1], const_value[6])
            integerDifferent_part_1_3, integerDifferent_qualified_3 = int_generator(mutated_operator_lists[2], const_value[6])
            integerDifferent_part_1_4, integerDifferent_qualified_4 = int_generator(mutated_operator_lists[3], const_value[6])
            integerDifferent_part_2_1, integerDifferent_qualified_5 = int_generator(mutated_operator_lists[4], const_value[7])
            integerDifferent_part_2_2, integerDifferent_qualified_6 = int_generator(mutated_operator_lists[5], const_value[7])
            integerDifferent_part_2_3, integerDifferent_qualified_7 = int_generator(mutated_operator_lists[6], const_value[7])
            integerDifferent_part_2_4, integerDifferent_qualified_8 = int_generator(mutated_operator_lists[7], const_value[7])
            integerDifferent_part_3_1, integerDifferent_qualified_9 = int_generator(mutated_operator_lists[8], const_value[8])
            integerDifferent_part_3_2, integerDifferent_qualified_10 = int_generator(mutated_operator_lists[9], const_value[8])
            integerDifferent_part_3_3, integerDifferent_qualified_11 = int_generator(mutated_operator_lists[10], const_value[8])
            integerDifferent_part_3_4, integerDifferent_qualified_12 = int_generator(mutated_operator_lists[11], const_value[8])
            integerDifferent_part_4_1, integerDifferent_qualified_13 = int_generator(mutated_operator_lists[12], const_value[9])
            integerDifferent_part_4_2, integerDifferent_qualified_14 = int_generator(mutated_operator_lists[13], const_value[9])
            integerDifferent_part_4_3, integerDifferent_qualified_15 = int_generator(mutated_operator_lists[14], const_value[9])
            integerDifferent_part_4_4, integerDifferent_qualified_16 = int_generator(mutated_operator_lists[15], const_value[9])

            operatorDifferent_part_1_1, operatorDifferent_qualified_1 = int_generator(mutated_operator_lists[16], const_value[10])
            operatorDifferent_part_1_2, operatorDifferent_qualified_2 = int_generator(mutated_operator_lists[17], const_value[10])
            operatorDifferent_part_1_3, operatorDifferent_qualified_3 = int_generator(mutated_operator_lists[18], const_value[10])
            operatorDifferent_part_1_4, operatorDifferent_qualified_4 = int_generator(mutated_operator_lists[19], const_value[10])
            operatorDifferent_part_2_1, operatorDifferent_qualified_5 = int_generator(mutated_operator_lists[20], const_value[11])
            operatorDifferent_part_2_2, operatorDifferent_qualified_6 = int_generator(mutated_operator_lists[21], const_value[11])
            operatorDifferent_part_2_3, operatorDifferent_qualified_7 = int_generator(mutated_operator_lists[22], const_value[11])
            operatorDifferent_part_2_4, operatorDifferent_qualified_8 = int_generator(mutated_operator_lists[23], const_value[11])
            operatorDifferent_part_3_1, operatorDifferent_qualified_9 = int_generator(mutated_operator_lists[24], const_value[12])
            operatorDifferent_part_3_2, operatorDifferent_qualified_10 = int_generator(mutated_operator_lists[25], const_value[12])
            operatorDifferent_part_3_3, operatorDifferent_qualified_11 = int_generator(mutated_operator_lists[26], const_value[12])
            operatorDifferent_part_3_4, operatorDifferent_qualified_12 = int_generator(mutated_operator_lists[27], const_value[12])
            operatorDifferent_part_4_1, operatorDifferent_qualified_13 = int_generator(mutated_operator_lists[28], const_value[13])
            operatorDifferent_part_4_2, operatorDifferent_qualified_14 = int_generator(mutated_operator_lists[29], const_value[13])
            operatorDifferent_part_4_3, operatorDifferent_qualified_15 = int_generator(mutated_operator_lists[30], const_value[13])
            operatorDifferent_part_4_4, operatorDifferent_qualified_16 = int_generator(mutated_operator_lists[31], const_value[13])


            random.shuffle(operator_value)
            # random.shuffle(reshuffled_operator_value)
        integer_list_1 = part_1_1 + part_1_2 + part_1_3 + part_1_4
        integer_list_2 = part_2_1 + part_2_2 + part_2_3 + part_2_4
        integer_list_3 = part_3_1 + part_3_2 + part_3_3 + part_3_4
        integer_list_4 = part_4_1 + part_4_2 + part_4_3 + part_4_4
        integer_list_5 = part_5_1 + part_5_2 + part_5_3 + part_5_4
        integer_list_6 = part_6_1 + part_6_2 + part_6_3 + part_6_4

        integerDifferent_integer_list_1 = integerDifferent_part_1_1 + integerDifferent_part_1_2 + integerDifferent_part_1_3 + integerDifferent_part_1_4
        integerDifferent_integer_list_2 = integerDifferent_part_2_1 + integerDifferent_part_2_2 + integerDifferent_part_2_3 + integerDifferent_part_2_4
        integerDifferent_integer_list_3 = integerDifferent_part_3_1 + integerDifferent_part_3_2 + integerDifferent_part_3_3 + integerDifferent_part_3_4
        integerDifferent_integer_list_4 = integerDifferent_part_4_1 + integerDifferent_part_4_2 + integerDifferent_part_4_3 + integerDifferent_part_4_4

        operatorDifferent_integer_list_1 = operatorDifferent_part_1_1 + operatorDifferent_part_1_2 + operatorDifferent_part_1_3 + operatorDifferent_part_1_4
        operatorDifferent_integer_list_2 = operatorDifferent_part_2_1 + operatorDifferent_part_2_2 + operatorDifferent_part_2_3 + operatorDifferent_part_2_4
        operatorDifferent_integer_list_3 = operatorDifferent_part_3_1 + operatorDifferent_part_3_2 + operatorDifferent_part_3_3 + operatorDifferent_part_3_4
        operatorDifferent_integer_list_4 = operatorDifferent_part_4_1 + operatorDifferent_part_4_2 + operatorDifferent_part_4_3 + operatorDifferent_part_4_4

    return operator_list, reshuffled_operator_list, integer_list_1, integer_list_2, integer_list_3, integer_list_4, integer_list_5, integer_list_6, \
        integerDifferent_integer_list_1, integerDifferent_integer_list_2, integerDifferent_integer_list_3, integerDifferent_integer_list_4, \
        operatorDifferent_integer_list_1, operatorDifferent_integer_list_2, operatorDifferent_integer_list_3, operatorDifferent_integer_list_4, \
        chosen_mutate_type_list


# Decide whether to display the constant on the center of each panel as hints
def whether_center(prob_type, geom_conditions):
    show_center = False
    if prob_type == "Composition":
        show_center = np.random.choice([True, False])
    elif prob_type == "Combination":
        condition_2 = geom_conditions[1]
        geom_relation = condition_2.grelation
        if geom_relation == "overlap" or geom_relation == "include":
            show_center = np.random.choice([True, False])
    return show_center


# Prune the sampled operators for number generation use
def operator_prune(num_blanks, operator_value):
    """
    Function to prune the operator from the operator_value list that will be used for number generation 
    and 50% chance to insert parentheses
    """
    insert_paren = 0
    operator_list = []
    if num_blanks == 1:
        operator_list.append(operator_value[0])
    else: # randomly introduce parentheses (only once, use insert_paren to mark it)
        for i in range(num_blanks):
            if operator_value[i] == "+" or operator_value[i] == "-":
                # If the operator is "+" or "-", check if either one of the operators on both sides is "*" or "/"
                if (i > 0 and (operator_value[i - 1] == "*" or operator_value[i - 1] == "/")) or \
                (i < len(operator_value) - 1 and (operator_value[i + 1] == "*" or operator_value[i + 1] == "/")):
                    # If conditions are met, randomly decide wherether to insert parentheses
                    if insert_paren == 0 and i < num_blanks - 1:
                        insert_paren = np.random.choice([0, 1])
            
            # After deciding to insert parentheses, insert them
            if insert_paren == 1:
                operator_list.extend(["(", operator_value[i], ")"])
                insert_paren = 2
            else:
                operator_list.append(operator_value[i])

    # print("In operator_prune func, Operator_list:", operator_list)    # ! For debugging
    return operator_list


def mutated_operator_prune(num_blanks, context_pruned_operator_list, chosen_mutate_type, opreator_indices_to_mutate):
    """
    Parameters:
        num_blanks: int
            Number of operators to be pruned in the expression, in simpler terms, the number of operators in the expression
    """
    verbose = False
    if verbose:
        print(("1. In mutated_operator_prune: Chosen_mutate_type:", chosen_mutate_type))                        # ! For debugging
        print(("2. In mutated_operator_prune: context_pruned_operator_list:", context_pruned_operator_list))    # ! For debugging

    assert chosen_mutate_type in ['MutOp', 'MutPar', 'MutBoth', 'MutFalse'], "Invalid mutate type"
    if chosen_mutate_type == 'MutOp' or chosen_mutate_type == 'MutBoth' or chosen_mutate_type == 'MutPar':
        mutated_operator_list = copy.deepcopy(context_pruned_operator_list) # Make a copy of the context_pruned_operator_list to mutate
        
        if chosen_mutate_type == 'MutOp':
            operators = ['+', '-', '*', '/']

            # Permutation combinations of the operators
            permutation_combinations = list(itertools.product(operators, repeat=len(opreator_indices_to_mutate)))

            operators_to_mutate = [context_pruned_operator_list[i] for i in opreator_indices_to_mutate]

            # Convert operators_to_mutate to a tuple to match the format of product_combinations
            operators_to_mutate_tuple = tuple(operators_to_mutate)

            # Remove the operators_to_mutate combination from product_combinations
            filtered_combinations = [comb for comb in permutation_combinations if comb != operators_to_mutate_tuple]

            # Mutate the operators in context_pruned_operator_list according to the index and one of the filtered_combinations
            selected_combination = random.choice(filtered_combinations)
            
            # Mutate the operators in context_pruned_operator_list according to the index and the selected combination
            for index, new_operator in zip(opreator_indices_to_mutate, selected_combination):
                mutated_operator_list[index] = new_operator

            if verbose:
                print("3. From mutate_type == 'MutOp'")                      # ! For debugging
                print(("4. Mutated_operator_list:", mutated_operator_list))    # ! For debugging

            assert context_pruned_operator_list != mutated_operator_list, "No mutation has been made"
            return mutated_operator_list

        if chosen_mutate_type == 'MutPar' or chosen_mutate_type == 'MutBoth':
            # Check if mutation for parentheses is applicable
            # Minimum requirements: 1. more than one operator, 2. at least one + OR - and at least one * OR /
            if num_blanks > 1 and ('+' in context_pruned_operator_list or '-' in context_pruned_operator_list) and ('*' in context_pruned_operator_list or '/' in context_pruned_operator_list):
                if verbose:
                    print("3a. Minumum requirement for adding parenthses is met.")    # ! For debugging

                # Now all can add parenthesis
                if num_blanks == 2 or (context_pruned_operator_list.count('+') + context_pruned_operator_list.count('-') == 1):
                    """
                    If there is only 2 number of blanks, we can only add 1 pair of parentheses
                    If there is only 1 + or - operator, we can only add 1 pair of parentheses
                    """
                    # Check if there are already parentheses in the context_pruned_operator_list for the two operators
                    if '(' not in context_pruned_operator_list and ')' not in context_pruned_operator_list:
                        # Add parentheses for the + or - operator
                        for i in range(len(context_pruned_operator_list)):
                            if context_pruned_operator_list[i] == '+' or context_pruned_operator_list[i] == '-':
                                # mutated_operator_list.extend(['(', mutated_operator_list[i], ')'])
                                mutated_operator_list.insert(i, '(')
                                mutated_operator_list.insert(i + 2, ')')

                        if verbose:
                            print("3. MutPar is carried out")                            # ! For debugging
                            print(("4. mutated_operator_list:", mutated_operator_list))    # ! For debugging
                    
                    else: # There is already parentheses, thereful MutOp is needed to be executed
                        # Randomly mutate one operator that is not a parentheses
                        operator_indices = [i for i, op in enumerate(mutated_operator_list) if op not in ['(', ')']]
                        random_operator_index_to_mutate = random.choice(operator_indices)
                        original_operator = mutated_operator_list[random_operator_index_to_mutate]
                        new_operator = random.choice([op for op in ['+', '-', '*', '/'] if op != original_operator])
                        mutated_operator_list[random_operator_index_to_mutate] = new_operator
                        
                        if verbose:
                            print("3. MutPar not applicable, already have parentheses, MutOp is carried out")    # ! For debugging
                            print(("4. mutated_operator_list:", mutated_operator_list))                            # ! For debugging
                    
                if num_blanks > 2 and (context_pruned_operator_list.count('+') + context_pruned_operator_list.count('-') == 2):
                    if not ('(' in context_pruned_operator_list and ')' in context_pruned_operator_list):
                        """
                        Condition:
                            1. Opertor more than 2
                            2. Only have 2 + or - operators
                            3. Initial solution does not contain parentheses
                        This specific situation we can add 1 or 2 pairs of parentheses, depending on the position of the operators,
                        """
                        # First, randomly choose whether to add one or two pairs of parentheses
                        num_parentheses = np.random.choice([1, 2])
                        parentheses_added = 0

                        for i in range(len(context_pruned_operator_list)):
                            if context_pruned_operator_list[i] == '+' or context_pruned_operator_list[i] == '-':
                                # Check if a * or / operator is adjacent to the current operator
                                if (i > 0 and context_pruned_operator_list[i - 1] in ['*', '/']) or \
                                (i < len(context_pruned_operator_list) - 1 and context_pruned_operator_list[i + 1] in ['*', '/']):
                                    # Insert parentheses around the + or - operator
                                    mutated_operator_list.insert(i, '(')
                                    mutated_operator_list.insert(i + 2, ')')
                                    parentheses_added += 1
                                    break

                        if (num_parentheses == parentheses_added) and verbose:
                            print(("3. MutPar with", parentheses_added))                  # ! For debugging
                            print(("4. mutated_operator_list:", mutated_operator_list))   # ! For debugging

                        if num_parentheses == 2:
                            """
                            The number of parentheses to add is 2, one pair of parentheses has already been added tp the mutated_operator_list
                            This situation will handle the nested parentheses needed to be added
                            There is e.g, a - b * c + d that no need nested
                            """
                            # Create a temporary list to store the mutated version
                            temp_mutated_operator_list = copy.deepcopy(mutated_operator_list)

                            # Iterate over the original mutated_operator_list
                            for i in range(len(mutated_operator_list)):
                                if mutated_operator_list[i] == '+' or mutated_operator_list[i] == '-':
                                    # Check if the operator is not already wrapped in parentheses
                                    if (i == 0 or mutated_operator_list[i - 1] != '(') and \
                                    (i == len(mutated_operator_list) - 1 or mutated_operator_list[i + 1] != ')'):
                                        if i != 0 and mutated_operator_list[i - 1] == ')':
                                            temp_mutated_operator_list.insert(i + 1, ')')
                                            # Find the position of the opening parenthesis
                                            opening_paren_index = i - 1
                                            while opening_paren_index >= 0 and temp_mutated_operator_list[opening_paren_index] != '(':
                                                opening_paren_index -= 1
                                            temp_mutated_operator_list.insert(opening_paren_index, '(')
                                            parentheses_added += 1
                                        elif i != len(mutated_operator_list) - 1 and mutated_operator_list[i + 1] == '(':
                                            if i > 0:
                                                temp_mutated_operator_list.insert(i - 1, '(')
                                            else:
                                                temp_mutated_operator_list.insert(0, '(')
                                            closing_paren_index = i + 1
                                            while closing_paren_index < len(temp_mutated_operator_list) and temp_mutated_operator_list[closing_paren_index] != ')':
                                                closing_paren_index += 1
                                            temp_mutated_operator_list.insert(closing_paren_index, ')')
                                            parentheses_added += 1
                                        else:
                                            temp_mutated_operator_list.insert(i, '(')
                                            temp_mutated_operator_list.insert(i + 2, ')')
                                            parentheses_added += 1

                            # After mutation, update mutated_operator_list with the changes from the temporary list
                            mutated_operator_list = temp_mutated_operator_list
                            
                            if (num_parentheses == parentheses_added) and verbose:
                                print(("3. MutPar with", parentheses_added))                  # ! For debugging
                                print(("4. mutated_operator_list:", mutated_operator_list))   # ! For debugging
                        
                        assert parentheses_added == num_parentheses, "Number of parentheses added does not match the randomly chosen number of parentheses to add"

                    elif ('(' in context_pruned_operator_list and ')' in context_pruned_operator_list):
                        """
                        Condition:
                            1. Opertor more than 2
                            2. Only have 2 + or - operators
                            3. Initial solution contain parentheses

                        This specific situation we need to check if initial solution already have parentheses.
                        In this situation we can only add one pair of parentheses
                        NOTE: In this situation we do not need to consider if they are beside * or / operators
                        NOTE: In this situation we do not need to consider if they are beside ) on left and ( on right
                        """
                        # Add the parentheses for the + or - operators that has not yet been enclosed in parentheses
                        for i in range(len(context_pruned_operator_list)):
                            if context_pruned_operator_list[i] == '+' or context_pruned_operator_list[i] == '-':
                                # Check if the operator is not already wrapped by parentheses
                                if (i == 0 or context_pruned_operator_list[i - 1] != '(') and \
                                (i == len(context_pruned_operator_list) - 1 or context_pruned_operator_list[i + 1] != ')'): #!!!
                                    if (i != 0 and context_pruned_operator_list[i - 1] == ')'):
                                        mutated_operator_list.insert(i + 1, ')')
                                        # Find the position of the closing parenthesis ')'
                                        opening_paren_index = i - 1
                                        while opening_paren_index >= 0 and mutated_operator_list[opening_paren_index] != '(':
                                            opening_paren_index -= 1
                                        mutated_operator_list.insert(opening_paren_index, '(')
                                    elif (i != len(context_pruned_operator_list) - 1 and context_pruned_operator_list[i + 1] == '('):
                                        if i > 0:
                                            mutated_operator_list.insert(i - 1, '(')
                                        else:
                                            mutated_operator_list.insert(0, '(')
                                        closing_paren_index = i + 1
                                        while closing_paren_index < len(mutated_operator_list) and mutated_operator_list[closing_paren_index] != ')':
                                            closing_paren_index += 1
                                        mutated_operator_list.insert(closing_paren_index, ')')
                                    else:
                                        # mutated_operator_list.extend(['(', mutated_operator_list[i], ')'])      
                                        mutated_operator_list.insert(i, '(')
                                        mutated_operator_list.insert(i + 2, ')')
                        if verbose:
                            print("3. MutPar for +1 pair is carried out")                # ! For debugging
                            print(("4. mutated_operator_list:", mutated_operator_list))    # ! For debugging

                # Check if that the total number of + and - is more than 2
                if num_blanks > 2 and (context_pruned_operator_list.count('+') + context_pruned_operator_list.count('-') > 2):
                    """
                    Condition: 
                        1. Minimum number of blanks is more than 2 
                        2. Total number of + and - operators is more than 2 (3 and more)
                        3. At least 1 * or / operator (Initial condition)

                    NOTE: When all above condition is met, we can add 1 or 2 pairs of parentheses even if initial pruned operator list already have parentheses
                    NOTE: The number of + and - corresponds to the number of pairs of parentheses that can be added, however only a minimum number of 1 * or / operator is needed

                    ! NOTE: In this situation we NEED to consider if they are beside * or / operators
                    ! MUST ALSO CONSIDER THAT THE + or - must be beside a * or - for the parenthesis to be effective or ) on left and ( on right
                    ! IT MUST BE EITHER BESIDE A PARENTHESES OR * or /.
                    """
                    # First, randomly choose whether to add one or two pairs of parentheses
                    num_parentheses = np.random.choice([1, 2])
                    parentheses_added = 0

                    while parentheses_added < num_parentheses:
                        if verbose:
                            print(("len(mutated_operator_list):", len(mutated_operator_list)))    # ! For debugging
                        
                        i = 0
                        while i <= len(mutated_operator_list) - 1:
                            if verbose:
                                print(("context_pruned_operator_list: ", mutated_operator_list))
                                print(("context_pruned_operator_list[i]", mutated_operator_list[i]))
                                print(("context_pruned_operator_list[i-1]", mutated_operator_list[i-1]))

                            if ((mutated_operator_list[i] == '+' or mutated_operator_list[i] == '-') \
                                and (i == 0 or i == (len(mutated_operator_list) - 1) or \
                                    (i > 0 and 
                                        (mutated_operator_list[i-1] != '(' and mutated_operator_list[i-1] != ')') or
                                        (mutated_operator_list[i+1] != '(' and mutated_operator_list[i+1] != ')')
                                    ) or \
                                    (i > 0 and 
                                        (mutated_operator_list[i-1] == ')' and mutated_operator_list[i+1] == '(')   # !!! THIS IS NEEDED FOR  the outer bracket ((+) + ( +)))
                                    )
                                    )
                                and (((i != 0 and mutated_operator_list[i - 1] == ')') or \
                                    (i != len(mutated_operator_list) - 1 and mutated_operator_list[i + 1] == '(')
                                    ) or 
                                    ((i > 0 and mutated_operator_list[i - 1] in ['*', '/']) or \
                                        (i < len(mutated_operator_list) - 1 and mutated_operator_list[i + 1] in ['*', '/'])
                                    )
                                    )
                            ):
                                """
                                Above if statement check if parentheses can be added.
                                Conditions:
                                    NOTE: Condition 1 2 and 3 are needed to be met
                                    1. Check whether the current operator is + or -
                                    2.1 It is index i = 0 or n - 1
                                        OR
                                    2.2 Basically just i only?: i > 0 AND not ['(', 'i'] or [')', 'i'] AND not ['i', ')'] or ['i', '('] 
                                        OR
                                    2.3 i > 0 AND IS ') i ('
                                    3.1 Basically to see if can add outer parenthesis: i != 0 and [')','i'] OR i != n - 1 and ['i','(']
                                        OR
                                    3.2 Basically i is beside * or /: i > 0 and ['* or /', 'i'] OR i < n - 1 and ['i+1', '*', '/'] 
                                """
                                insert_parentheses = np.random.choice([0, 1])
                                if insert_parentheses == 1:
                                    if ((i != 0 and i != len(mutated_operator_list) - 1 and mutated_operator_list[i - 1] == ')' and mutated_operator_list[i + 1] != '(')
                                        or (i == 0 and mutated_operator_list[i + 1] != '(')
                                        or (i == len(mutated_operator_list) - 1 and mutated_operator_list[i - 1] == ')')):
                                        """
                                        This part add i ) and wraped for '(' 
                                        """
                                        mutated_operator_list.insert(i + 1, ')')
                                        # Find the position of the closing parenthesis ')'
                                        opening_paren_index = i
                                        if opening_paren_index == 0:
                                            mutated_operator_list.insert(i, '(')
                                        else:
                                            opening_paren_index = i - 1
                                            while opening_paren_index > 0 and mutated_operator_list[opening_paren_index] != '(':
                                                opening_paren_index -= 1
                                            mutated_operator_list.insert(opening_paren_index, '(')
                                        parentheses_added += 1

                                        if verbose:
                                            print("break in 1")

                                        break                                
                                    elif ((i != 0 and i != len(mutated_operator_list) - 1 and mutated_operator_list[i + 1] == '(' and mutated_operator_list[i - 1] != ')')
                                        or (i == len(mutated_operator_list) - 1 and mutated_operator_list[i - 1] != ')')
                                        or (i == 0 and mutated_operator_list[i + 1] == '(')):
                                        """
                                        This part add ( i and wraped for ')'
                                        """
                                        if i > 0:
                                            mutated_operator_list.insert(i, '(')
                                        else:
                                            mutated_operator_list.insert(0, '(')
                                        closing_paren_index = i + 1
                                        while closing_paren_index < len(mutated_operator_list) and mutated_operator_list[closing_paren_index] != ')':
                                            closing_paren_index += 1
                                        mutated_operator_list.insert(closing_paren_index, ')')
                                        parentheses_added += 1

                                        if verbose:
                                            print("break in 2")

                                        break

                                    elif (i != 0 and mutated_operator_list[i - 1] == ')' and i != len(mutated_operator_list) - 1 and mutated_operator_list[i + 1] == '('):
                                        """
                                        when parenthesis on both side
                                        This part add wraped '( )' on both side
                                        """
                                        # Handle case when there are parentheses on both sides
                                        opening_paren_index = i - 1
                                        while opening_paren_index >= 0 and mutated_operator_list[opening_paren_index] != '(':
                                            opening_paren_index -= 1
                                        mutated_operator_list.insert(opening_paren_index, '(')
                                        closing_paren_index = i + 1
                                        while closing_paren_index < len(mutated_operator_list) and mutated_operator_list[closing_paren_index] != ')':
                                            closing_paren_index += 1
                                        mutated_operator_list.insert(closing_paren_index, ')')
                                        parentheses_added += 1

                                        if verbose:
                                            print("break in 4 special both")

                                        break

                                    # Case for not nested parentheses
                                    if ((i > 0 and mutated_operator_list[i - 1] in ['*', '/']) or \
                                        (i < len(mutated_operator_list) - 1 and mutated_operator_list[i + 1] in ['*', '/'])):
                                        mutated_operator_list.insert(i, '(')
                                        mutated_operator_list.insert(i + 2, ')')
                                        parentheses_added += 1
                                        if verbose:
                                            print("break in 3")

                                        break
                            i += 1
                    if verbose:
                        print(("3a1. mutated_operator_list after add parenthesis:", mutated_operator_list))    # ! For debugging
            
                if chosen_mutate_type == 'MutBoth':
                    """
                    This condition is when parentheses are already added.
                    Task left is only to mutate the operators.
                    """
                    temp_mutated_operator_list = copy.deepcopy(context_pruned_operator_list)    # Make a copy of the context_pruned_operator_list to mutate

                    operators = ['+', '-', '*', '/']

                    # Permutation combinations of the operators
                    permutation_combinations = list(itertools.product(operators, repeat=len(opreator_indices_to_mutate)))

                    operators_to_mutate = [context_pruned_operator_list[i] for i in opreator_indices_to_mutate]

                    # Convert operators_to_mutate to a tuple to match the format of product_combinations
                    operators_to_mutate_tuple = tuple(operators_to_mutate)

                    # Remove the operators_to_mutate combination from product_combinations
                    filtered_combinations = [comb for comb in permutation_combinations if comb != operators_to_mutate_tuple]

                    # Mutate the operators in context_pruned_operator_list according to the index and one of the filtered_combinations
                    selected_combination = random.choice(filtered_combinations)
                    
                    # Mutate the operators in context_pruned_operator_list according to the index and the selected combination
                    for index, new_operator in zip(opreator_indices_to_mutate, selected_combination):
                        temp_mutated_operator_list[index] = new_operator

                    if verbose:
                        # NOTE here mutated_operator_list is still a copy of original context_pruned_operator_list
                        print(("Context pruned operator list:", context_pruned_operator_list))    # ! For debugging
                        print(("Temp mutated operator list:", temp_mutated_operator_list))        # ! For debugging
                        print(("Original mutated operator list:", mutated_operator_list))         # ! For debugging

                    # Replace the operator in mutated_operator_list with the operator in temp_mutated_operator_list ignoring the added parenthesis
                    temp_index = 0
                    for i in range(len(mutated_operator_list)):
                        if mutated_operator_list[i] not in ['(', ')']:
                            while temp_mutated_operator_list[temp_index] in ['(', ')']:
                                temp_index += 1
                            mutated_operator_list[i] = temp_mutated_operator_list[temp_index]
                            temp_index += 1
                    
                    if verbose:
                        print(("Mutated operator list after replacing with mutated operator:", mutated_operator_list))    # ! For debugging
                        print("3. From mutate_type == 'MutBoth'")                    # ! For debugging
                        print(("4. mutated_operator_list:", mutated_operator_list))    # ! For debugging

                    assert context_pruned_operator_list != mutated_operator_list, "No mutation has been made"
                    return mutated_operator_list
                
                if verbose:
                    print("3. From mutate_type == 'MutPar'")
                    print(("4. mutated_operator_list:", mutated_operator_list))    # ! For debugging

                assert context_pruned_operator_list != mutated_operator_list, "No mutation has been made"
                return mutated_operator_list
            
            else:
                """
                Mutation for if parentheses is not applicable
                Default to mutate the operators
                """
                operators = ['+', '-', '*', '/']

                # Permutation combinations of the operators
                permutation_combinations = list(itertools.product(operators, repeat=len(opreator_indices_to_mutate)))

                if verbose:
                    print(("len(opreator_indices_to_mutate):", len(opreator_indices_to_mutate)))    # ! For debugging
                    print(("permutation_combinations:", permutation_combinations))                  # ! For debugging
                    print(("opreator_indices_to_mutate:", opreator_indices_to_mutate))              # ! For debugging

                operators_to_mutate = [context_pruned_operator_list[i] for i in opreator_indices_to_mutate]

                # Convert operators_to_mutate to a tuple to match the format of product_combinations
                operators_to_mutate_tuple = tuple(operators_to_mutate)

                # Remove the operators_to_mutate combination from product_combinations
                filtered_combinations = [comb for comb in permutation_combinations if comb != operators_to_mutate_tuple]

                # Mutate the operators in context_pruned_operator_list according to the index and one of the filtered_combinations
                selected_combination = random.choice(filtered_combinations)
                
                # Mutate the operators in context_pruned_operator_list according to the index and the selected combination
                for index, new_operator in zip(opreator_indices_to_mutate, selected_combination):
                    mutated_operator_list[index] = new_operator
                
                if verbose:
                    print("3. From mutate_type == 'Default to MutOp'")            # ! For debugging
                    print(("4. mutated_operator_list:", mutated_operator_list))     # ! For debugging

                assert context_pruned_operator_list != mutated_operator_list, "No mutation has been made"
                return mutated_operator_list

        if verbose:
            print("3. From mutate_type == 'FAILED'. IT SHOULD NOT REACH HERE")    # ! For debugging
            print(("4. mutated_operator_list:", mutated_operator_list))             # ! For debugging

        raise RuntimeError("Unexpected state: mutate_type == 'FAILED'. Execution should not reach here.")
        assert context_pruned_operator_list != mutated_operator_list, "No mutation has been made"
        return mutated_operator_list
    
    elif chosen_mutate_type == 'MutFalse':
        if verbose:
            print("3. From mutate_type == 'MutFalse'")

        return context_pruned_operator_list


def int_generator(operator_list, const_value):
    """
    Generate the numbers based on a calculation process defined by the given operators and constants
    """
    filled_expression = fill_expression(operator_list)  # Fill in the operator list to with expressions (a, b, c, d, ...)
    after_expression = middle_to_after(filled_expression)
    node_list, levels = build_calculator_tree(after_expression)

    new_node_list, integer_list, qualified = number_sampler(node_list, const_value, levels)

    # document the expression for calculation
    final_expression = str(integer_list[0])
    previous_index = 0
    previous_num = str(integer_list[previous_index])
    for i in range(len(operator_list)):
        if operator_list[i] == "(" and i == 0:
            final_expression = "(" + previous_num
        elif i < len(operator_list) - 1 and operator_list[i+1] == "(":
            final_expression += operator_list[i]
        elif operator_list[i] == ")":
            final_expression += ")"
        else:
            previous_index += 1
            previous_num = str(integer_list[previous_index])
            final_expression += operator_list[i] + previous_num
    final_expression += "=" + str(const_value)

    verbose = False
    if verbose:
        print(("In int_generator func, Final expression:", final_expression))    # ! For debugging
        print(("In int_generator func, Qualified:", qualified))                  # ! For debugging
    
    return integer_list, qualified


def disp_num(img, int_list, pos_list, blank, font_size, show_center, chosen_mutate_type = None):
    """
    Display the numbers on the panels and save the answer
    """
    num_pos = len(pos_list)
    mark_position = "none"
    answer = 0

    from PIL import Image
    img = Image.fromarray(img)
            
    draw = ImageDraw.Draw(img)
    draw.fontmode = "1"
    # Load default font
    fnt = ImageFont.load_default()

    STROKE_WIDTH = 0
    ALIGN = "center"

    blank = True
    if blank:   # If blank is True, mutation is enabled
        mark = -1    # Initialize the marked position to -1
        marks = -1   # Initialize the marked position to -1
        mark_positions = []

        if show_center:
            # If center is shown, exclude the center position from the random choice
            #NOTE: if center is shown you need to consider wheter to mutate centre or not
            mark = np.random.choice(list(range(0, num_pos - 1))) # OR             mark = np.random.choice(range(0, num_pos))
            mark_position = mark
        else:
            if chosen_mutate_type == 'MutFalse':
                # If the mutation type is MutFalse, randomly choose the number of position and position to mutate
                num_to_mutate = np.random.choice([1, 2])        # Choose whether to mutate 1 or 2 positions.
                marks = np.random.choice(list(range(0, num_pos)), size=num_to_mutate, replace=False)  # Randomly select positions.
                mark_positions = marks.tolist()  # Convert NumPy array to Python list


        # Display the numbers and the question mark
        for i in range(0, num_pos):
            (x, y) = pos_list[i]
            if i not in mark_positions:
                """
                If i is not the marked position, display the number normally
                """
                text = str(int_list[i])
                if len(text) >= 2:
                    draw.text((x-5, y-5), text, font=fnt, fill=0)
                else:
                    draw.text((x-3, y-5), text, font=fnt, fill=0)

            else:  
                """
                If i in mark_positions and mutatetype = MutFalse, mutate the number to a wrong number
                """
                mutated_value_to_display = mutate_constant(int_list[i])

                text = str(mutated_value_to_display)  

                if len(text) >= 2:
                    draw.text((x-5, y-5), text, font=fnt, fill=0) # currently perfect
                else:
                    draw.text((x-3, y-5), text, font=fnt, fill=0)

                answer = int_list[i]    # Save the marked number as the answer

    else:
        for i in range(0, num_pos):
            (x, y) = pos_list[i]
            
            text = str(int_list[i])
            if len(text) >= 2:
                draw.text((x-5, y-5), text, font=fnt, fill=0) # currently perfect    
            else:
                draw.text((x-3, y-5), text, font=fnt, fill=0)

    mark_position = -1
    return answer, img, mark_position


# permutate the integer list to render its order as the clockwise order of the integers appeared on the panel
def reverse_arrange(int_list, mode, part):
    """
    Permutate the integer list to render its order as the clockwise order of the integers appeared on the panel
    For the return list, so its in the correct order, but the image is already generated, so using this func shud not affect the image.
    """
    num = len(int_list)
    new_list = []
    if mode == 2:
        if part == 2:
            list_1_1, list_1_2 = int_list[0: num // 2], int_list[num // 2: num]
            for i in range(len(list_1_1)):
                new_list.extend([list_1_1[i], list_1_2[i]])
        elif part == 3:
            list_1_1, list_1_2, list_1_3 = int_list[0: num // 3], int_list[num // 3: 2 * num // 3], int_list[2 * num // 3: num]
            for i in range(len(list_1_1)):
                new_list.extend([list_1_1[i], list_1_2[i], list_1_3[i]])
        elif part == 4:
            list_1_1, list_1_2 = int_list[0: num // 4], int_list[num // 4: num // 2]
            list_1_3, list_1_4 = int_list[num // 2: 3 * num // 4], int_list[3 * num // 4: num]
            for i in range(len(list_1_1)):
                new_list.extend([list_1_1[i], list_1_2[i], list_1_3[i], list_1_4[i]])
    elif mode == 3:
        new_list, list_1_2 = int_list[0: num // 2], int_list[num // 2: num]
        if num == 8:
            new_list.extend([list_1_2[3], list_1_2[2], list_1_2[1], list_1_2[0]])
        elif num == 6:
            new_list.extend([list_1_2[2], list_1_2[1], list_1_2[0]])
        elif num == 4:
            new_list.extend([list_1_2[1], list_1_2[0]])
    elif mode == 4 and num == 8:
        list_1_1, list_1_2 = int_list[0: num // 2], int_list[num // 2: num]
        new_list = [list_1_1[0], list_1_2[0], list_1_1[3], list_1_2[3], list_1_1[1], list_1_2[1], list_1_1[2], list_1_2[2]]

    return new_list