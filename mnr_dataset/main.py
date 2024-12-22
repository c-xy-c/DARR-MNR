# -*- coding: utf-8 -*-
import numpy as np
from tqdm import trange
from const import PANEL_SIZE, SEED_VALUE, VERBOSE
from Trees import (combination_prob, composition_prob, partition_prob, prune_problem_tree)
from Drawing import (imshow, imsave, drawing_panels, rendering_panels, rendering_one_panel)
from Num_Arrange import reverse_arrange
import os
# import cv2
import random
import argparse
np.random.seed(SEED_VALUE)
random.seed(SEED_VALUE)

# Parse arguments
parser = argparse.ArgumentParser(description="Problem Visualization")
parser.add_argument('--visualize', type=int, choices=[0, 1, 2], default=0,
                    help="Visualization mode: 0 for no visualization (default), 1 for combined panels, 2 for a single panel.")
parser.add_argument('--num_prob', type=int, default=1000, help="Number of problems to be generated.")
FLAGS = parser.parse_args()

VISUALIZE = FLAGS.visualize
NUM_PROB = FLAGS.num_prob

assert VISUALIZE in [0, 1, 2], "Invalid visualization mode"
assert NUM_PROB > 0, "Invalid number of problems"


# generate problems that pruned by given parameters
def generate_problem(problem_configs, num_of_problem, key, geom_type, geom_condition, interpretation, analytical_part, seperate):
    if VERBOSE:
        ("Geom_type: %s " % geom_type)
    for n in trange(num_of_problem):
        count = n % 10
        if count in [0, 2, 4, 6, 8, 9]:
            set_name = "train_set"
        elif count in [1, 7]:
            set_name = "val_set"
        else:
            set_name = "test_set"

        root = problem_configs[key] # key = "composition_problem", "combination_problem", "partition_problem", problem_configs should be correspond to the tree in tree.py, refer 'problems' in def main
        new_root = root.sample()    # sample() method returns a list with a specified number of randomly selected items from a sequence
        pruned_root = prune_problem_tree(new_root, geom_type, geom_condition, interpretation, analytical_part)

        integerDifferent_answers, operator, list_1, list_2, list_3, list_4, list_5, list_6, \
            integerDifferent_list_1, integerDifferent_list_2, integerDifferent_list_3, integerDifferent_list_4, \
            operatorDifferent_list_1, operatorDifferent_list_2, operatorDifferent_list_3, operatorDifferent_list_4, \
            center, interpret, mode, part, integerDifferent_marks, images, wrong_answer_set_images = drawing_panels(pruned_root)
        
        prob_type, conditions = pruned_root.prepare()
        geometry = [conditions[0], conditions[1]]
        mathematics = [conditions[2], conditions[3]]

        if prob_type == "Combination":
            gcondition1 = geometry[0].type.get_value()
            gcondition2 = geometry[1].grelation.get_value()
        elif prob_type == "Composition":
            gcondition1 = geometry[0].type.get_value()
            gcondition2 = geometry[1].format.get_value()
        elif prob_type == "Partition":
            gcondition1 = geometry[0].type.get_value()
            gcondition2 = geometry[1].part.get_value()
        geometrial_conditions = [gcondition1, gcondition2]
        
        integerDifferent_list_1[integerDifferent_marks[0]] = "mark"
        integerDifferent_list_2[integerDifferent_marks[1]] = "mark"
        integerDifferent_list_3[integerDifferent_marks[2]] = "mark"
        integerDifferent_list_4[integerDifferent_marks[3]] = "mark"

        # Remove the last element of the images list
        correct_image_panel = images.pop()
        correct_answer_image_index = None

        random_selection = True
        if(random_selection):
            # Randomly choose 7 wrong panels and add into wrong answer_set_images list
            random_index_list = np.random.choice(range(0, len(wrong_answer_set_images)), 7, replace=False)

            if VERBOSE:
                print("random_index_list: %s" % random_index_list)  # For debugging purpose
            
            # Select the wrong images based on the random indices generated
            answer_set_images = [wrong_answer_set_images[i] for i in random_index_list]
            
            # Add the correct image to the list
            answer_set_images.append(correct_image_panel)
            
            # Shuffle the answer set images
            random.shuffle(answer_set_images)
            
            # Find the index of the correct image in the shuffled list
            correct_answer_image_index = answer_set_images.index(correct_image_panel)

            if VERBOSE:
                print("correct_index: %s" % correct_answer_image_index)  # For debugging purpose
                                
        else:
            answer_set_images = [wrong_answer_set_images[0],
                                wrong_answer_set_images[1],
                                wrong_answer_set_images[2],
                                wrong_answer_set_images[3],
                                wrong_answer_set_images[4],
                                wrong_answer_set_images[5],
                                wrong_answer_set_images[6],
                                correct_image_panel]

        target = correct_image_panel

        # Convert each Image object in the list to a numpy array
        images = [np.array(img) for img in images[:3]]
        answer_set_images = [np.array(img) for img in answer_set_images]

        # Now you can convert the entire list to a numpy array
        images = np.array(images)
        answer_set_images = np.array(answer_set_images)
        
        gestalt_law = "none"
        if VERBOSE:
            print("In main.py, mode: %s" % mode)

        if mode == 1:
            gestalt_law = "proximity"
        elif mode == 2:
            gestalt_law = "symmetry"

        if seperate:
            directory_path = "ProbSet/{0}/{1}".format(key, set_name)
            file_path = "{0}/prob_{1}_{2}_{3}_{4}_{5}.npz".format(directory_path, n, geom_type, geom_condition, interpretation, analytical_part)
            # file_path = "{0}/prob_{1}_{2}_{3}_{4}_{5}_{6}.npz".format(directory_path, n, geom_type, geom_condition, interpretation, analytical_part, gestalt_law)
        else:
            directory_path = "ProbSet/{0}".format(set_name)
            file_path = "{0}/prob_{1}_{2}_{3}_{4}_{5}_{6}.npz".format(directory_path, n, key, geom_type, geom_condition, interpretation, analytical_part)
            # file_path = "{0}/prob_{1}_{2}_{3}_{4}_{5}_{6}_{7}.npz".format(directory_path, n, key, geom_type, geom_condition, interpretation, analytical_part, gestalt_law)
 
        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(directory_path):
            try:
                os.makedirs(directory_path)
            except OSError as e:
                if e.errno != os.errno.EEXIST:
                    raise
        
        if VERBOSE:
            print("In main.py, File_path: %s" % file_path) # For debugging purpose

        # Save the problem panel, answer set panel and other attributes into a .npz file
        np.savez(file_path,
                context_images=images, 
                answer_set_images=answer_set_images, 
                correct_answer_image_index=correct_answer_image_index  # correct panel, currently storing the whole image
        )


        # To visualize the problem
        if VISUALIZE == 1:
            problem_image = rendering_panels(images[0], images[1], images[2],
                                            answer_set_images[0], answer_set_images[1], answer_set_images[2], answer_set_images[3],
                                            answer_set_images[4], answer_set_images[5], answer_set_images[6], answer_set_images[7], PANEL_SIZE)
            imsave(problem_image, "ProbSet/{}/prob_{}_{}_{}_{}_{}_{}.jpg".format(set_name, n, key, geom_type, geom_condition, interpretation, analytical_part))
            # imsave(problem_image, "ProbSet/{}/prob_{}_{}_{}_{}_{}_{}_{}.jpg".format(set_name, n, key, geom_type, geom_condition, interpretation, analytical_part, gestalt_law))

        elif VISUALIZE == 2:
            problem_image = rendering_one_panel(images[0], PANEL_SIZE)
            imsave(problem_image, "ProbSet/{}/prob_{}_{}_{}_{}_{}_{}.jpg".format(set_name, n, key, geom_type, geom_condition, interpretation, analytical_part))
            # imsave(problem_image, "ProbSet/{}/prob_{}_{}_{}_{}_{}_{}_{}.jpg".format(set_name, n, key, geom_type, geom_condition, interpretation, analytical_part, gestalt_law))

# generate the dataset containing equal amount of problems on all possible conditions
def generation(problem_configs, num_of_problem):
    for key in problem_configs.keys():
        pruned_types = ["circle", "square", "triangle", "rectangle", "hexagon"] # geometric shape types = ["circle", "square", "triangle", "rectangle", "hexagon"]
        if key == "combination_problem":
            geom_conditions = ["overlap", "include", "tangent"]
            for geom_condition in geom_conditions:
                for pr_type in pruned_types:
                    if VERBOSE:
                        print("pr_type: %s" % pr_type)

                    if geom_condition == "include" or (geom_condition == "overlap" and pr_type in ["rectangle", "triangle", "hexagon", "circle", "square"]):
                        interpretations = ["holistic", "analytical"]
                    else:
                        interpretations = ["holistic"]
                    for interpretation in interpretations:
                        if interpretation == "analytical":
                            if geom_condition == "include":
                                if pr_type == "triangle":
                                    analytical_parts = [2, 3]
                                else:
                                    analytical_parts = [2, 4]
                            elif geom_condition == "overlap":
                                if pr_type == "triangle":
                                    analytical_parts = [2]
                                elif pr_type in ["rectangle", "hexagon", "circle", "square"]:
                                    analytical_parts = [2]
                        else:
                            analytical_parts = [0]
                        for analytical_part in analytical_parts:
                            generate_problem(problem_configs, num_of_problem, key, pr_type, geom_condition, interpretation, analytical_part, False)

        elif key == "composition_problem":
            geom_conditions = ["line", "cross", "triangle", "square", "circle"]
            for geom_condition in geom_conditions:
                for pr_type in pruned_types:
                    if VERBOSE:
                        print("pr_type: %s" % pr_type)

                    interpretations = ["holistic", "analytical"]
                    for interpretation in interpretations:
                        if interpretation == "analytical":
                            if geom_condition == "triangle":
                                analytical_parts = [2, 3]
                            elif geom_condition == "line":
                                analytical_parts = [2]
                            else:
                                analytical_parts = [2, 4]
                        else:
                            analytical_parts = [0]
                        for analytical_part in analytical_parts:
                            generate_problem(problem_configs, num_of_problem, key, pr_type, geom_condition, interpretation, analytical_part, False)

        elif key == "partition_problem":
            geom_conditions = [2, 4, 6, 8]
            for geom_condition in geom_conditions:
                for pr_type in pruned_types:
                    if VERBOSE:
                        print("pr_type: %s" % pr_type)

                    if geom_condition == 2:
                        interpretations = ["holistic"]
                    else:
                        interpretations = ["holistic", "analytical"]
                    for interpretation in interpretations:
                        if interpretation == "analytical":
                            if geom_condition == 4:
                                analytical_parts = [2]
                            elif geom_condition == 6:
                                analytical_parts = [2, 3]
                            elif geom_condition == 8:
                                analytical_parts = [2, 4]
                        else:
                            analytical_parts = [0]
                        for analytical_part in analytical_parts:
                            generate_problem(problem_configs, num_of_problem, key, pr_type, geom_condition, interpretation, analytical_part, False)


def main():
    problems = {"composition_problem": composition_prob(),
                "combination_problem": combination_prob(),
                "partition_problem": partition_prob()}
    """
    Number of problems to be generated for Machine Number Reasoning (MNR) dataset
    - every 10 problems, 6 problems are for training, 2 problems are for validation, and 2 problems are for testing
    """
    num_prob = NUM_PROB
    generation(problems, num_prob)


if __name__ == "__main__":
    main()