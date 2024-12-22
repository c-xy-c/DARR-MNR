# -*- coding: utf-8 -*-

import cv2
import math
import copy
import numpy as np
from PIL import Image
from AOT import Root
from const import (PANEL_SIZE, CENTER, CENTER_1_1, CENTER_1_2, DEFAULT_WIDTH, LENGTH_1, LENGTH_2, LENGTH_3, SEED_VALUE)
from Num_Arrange import (disp_num, math_parser, whether_center)
np.random.seed(SEED_VALUE)

'''
How to draw panels: According to the sampled parameters about the problem, the function for each problem ----
draw_combination(), draw_composition(), draw_partition() will draw the geometrical figures and return a list of
coordinates to place the numbers. Then display numbers and question mark on these given positions.
'''


def imshow(array):
    image = Image.fromarray(array)
    image.show()


def imsave(array, file_path):
    image = Image.fromarray(array)
    image.save(file_path)


def rendering_panels(img1, img2, img3, 
                     answer_set_img1, answer_set_img2, answer_set_img3, answer_set_img4,
                     answer_set_img5, answer_set_img6, answer_set_img7, answer_set_img8, panel_size):
    """
    Function to render the all the panels into a single image for visualization purposes
    """
    # Create a grayscale image
    image = 255 * np.ones((3*panel_size + 2, 4*panel_size + 3), np.uint8)  # Add 2 for the border lines

    # Insert images
    image[0: panel_size, 0: panel_size] = img1
    image[0: panel_size, panel_size + 1: 2*panel_size + 1] = img2
    image[0: panel_size, 2*panel_size + 2: 3*panel_size + 2] = img3

    # Insert answer set images
    image[panel_size + 1: 2*panel_size + 1, 0: panel_size] = answer_set_img1
    image[panel_size + 1: 2*panel_size + 1, panel_size + 1: 2*panel_size + 1] = answer_set_img2
    image[panel_size + 1: 2*panel_size + 1, 2*panel_size + 2: 3*panel_size + 2] = answer_set_img3
    image[panel_size + 1: 2*panel_size + 1, 3*panel_size + 2: 4*panel_size + 2] = answer_set_img4

    image[2*panel_size + 2: 3*panel_size + 2, 0: panel_size] = answer_set_img5
    image[2*panel_size + 2: 3*panel_size + 2, panel_size + 1: 2*panel_size + 1] = answer_set_img6
    image[2*panel_size + 2: 3*panel_size + 2, 2*panel_size + 2: 3*panel_size + 2] = answer_set_img7
    image[2*panel_size + 2: 3*panel_size + 2, 3*panel_size + 2: 4*panel_size + 2] = answer_set_img8

    # Insert border lines
    image[panel_size:panel_size + 1, :] = 0
    image[2*panel_size + 1:2*panel_size + 2, :] = 0

    image[:, panel_size:panel_size + 1] = 0
    image[:, 2*panel_size + 1:2*panel_size + 2] = 0
    image[:, 3*panel_size + 2:3*panel_size + 3] = 0

    return image


def rendering_one_panel(img, panel_size):
    """
    Function to render the one panels into a single image for visualization purposes
    """
    # Create a grayscale image
    image = 255 * np.ones((panel_size, panel_size), np.uint8)

    # Insert img1
    image[0: panel_size, 0: panel_size] = img

    return image


def drawing_panels(root):
    # Decompose the panel into layout(geometrical) and algebra(mathematical) parts
    assert isinstance(root, Root)
    prob_type, conditions = root.prepare()  # !
    geom_conditions = [conditions[0], conditions[1]]
    math_conditions = [conditions[2], conditions[3]]
    # draw the geometrical figures for each type of problem
    font_size = 0.002 * PANEL_SIZE * 2
    if prob_type == "Combination":
        img, positions = draw_combination(geom_conditions)
    elif prob_type == "Composition":
        img, positions = draw_composition(geom_conditions)
        font_size = (1.0/600) * PANEL_SIZE * 1 # In python 2.7, to get float after division, it must be 1.0 instead of 1
    elif prob_type == "Partition":
        img, positions = draw_partition(geom_conditions)

    # Arrange and display the numbers on the panel
    # show_center = whether_center(prob_type, geom_conditions)
    show_center = False # Set to False to not show the hint in the centre and increase the difficulty
    interpret, mode, part, operator_list, reshuffled_operator_list, \
    int_list_1, int_list_2, int_list_3, int_list_4, int_list_5, int_list_6, \
    integerDifferent_int_list_1, integerDifferent_int_list_2, integerDifferent_int_list_3, integerDifferent_int_list_4, \
    operatorDifferent_int_list_1, operatorDifferent_int_list_2, operatorDifferent_int_list_3, operatorDifferent_int_list_4, \
    pos_list, chosen_mutate_type_list = math_parser(math_conditions,
                            geom_conditions,
                            prob_type,
                            positions,
                            show_center)

    img1, img2, img3, img4, img5, img6 = copy.deepcopy(img), copy.deepcopy(img), copy.deepcopy(img), copy.deepcopy(img), copy.deepcopy(img), copy.deepcopy(img)
    integerDifferent_img1, integerDifferent_img2, integerDifferent_img3, integerDifferent_img4 = copy.deepcopy(img), copy.deepcopy(img), copy.deepcopy(img), copy.deepcopy(img)
    operatorDifferent_img1, operatorDifferent_img2, operatorDifferent_img3, operatorDifferent_img4 = copy.deepcopy(img), copy.deepcopy(img), copy.deepcopy(img), copy.deepcopy(img)
    
    answer_1, img_1, mark_1 = disp_num(img1, int_list_1, pos_list, False, font_size, show_center)
    answer_2, img_2, mark_2 = disp_num(img2, int_list_2, pos_list, False, font_size, show_center)
    answer_3, img_3, mark_3 = disp_num(img3, int_list_3, pos_list, False, font_size, show_center) # Display the question mark
    answer_4, img_4, mark_4 = disp_num(img4, int_list_4, pos_list, False, font_size, show_center)
    answer_5, img_5, mark_5 = disp_num(img5, int_list_5, pos_list, False, font_size, show_center)
    answer_6, img_6, mark_6 = disp_num(img6, int_list_6, pos_list, False, font_size, show_center)

    integerDiffernt_answer_1, integerDifferent_img_1, integerDifferent_mark_1 = disp_num(integerDifferent_img1, integerDifferent_int_list_1, pos_list, True, font_size, show_center, chosen_mutate_type_list[0])
    integerDiffernt_answer_2, integerDifferent_img_2, integerDifferent_mark_2 = disp_num(integerDifferent_img2, integerDifferent_int_list_2, pos_list, True, font_size, show_center, chosen_mutate_type_list[1])
    integerDiffernt_answer_3, integerDifferent_img_3, integerDifferent_mark_3 = disp_num(integerDifferent_img3, integerDifferent_int_list_3, pos_list, True, font_size, show_center, chosen_mutate_type_list[2])
    integerDiffernt_answer_4, integerDifferent_img_4, integerDifferent_mark_4 = disp_num(integerDifferent_img4, integerDifferent_int_list_4, pos_list, True, font_size, show_center, chosen_mutate_type_list[3])
    
    operatorDiffernt_answer_1, operatorDifferent_img_1, operatorDifferent_mark_1 = disp_num(operatorDifferent_img1, operatorDifferent_int_list_1, pos_list, True, font_size, show_center, chosen_mutate_type_list[4])
    operatorDiffernt_answer_2, operatorDifferent_img_2, operatorDifferent_mark_2 = disp_num(operatorDifferent_img2, operatorDifferent_int_list_2, pos_list, True, font_size, show_center, chosen_mutate_type_list[5])
    operatorDiffernt_answer_3, operatorDifferent_img_3, operatorDifferent_mark_3 = disp_num(operatorDifferent_img3, operatorDifferent_int_list_3, pos_list, True, font_size, show_center, chosen_mutate_type_list[6])
    operatorDiffernt_answer_4, operatorDifferent_img_4, operatorDifferent_mark_4 = disp_num(operatorDifferent_img4, operatorDifferent_int_list_4, pos_list, True, font_size, show_center, chosen_mutate_type_list[7])

    integerDifferent_marks = [integerDifferent_mark_1, integerDifferent_mark_2, integerDifferent_mark_3, integerDifferent_mark_4]
    
    prob_answers = [integerDiffernt_answer_1, integerDiffernt_answer_2, integerDiffernt_answer_3, integerDiffernt_answer_4]
    prob_operator = operator_list
    prob_answer_set_reshuffled_operator = reshuffled_operator_list  # ! NOT USED and should not be used
    prob_images = [img_1, img_2, img_3, img_4, img_5, img_6]   # Store the images of the panels in a list
    prob_answer_set_images = [integerDifferent_img_1, integerDifferent_img_2, integerDifferent_img_3, integerDifferent_img_4,
                                operatorDifferent_img_1, operatorDifferent_img_2, operatorDifferent_img_3, operatorDifferent_img_4]  # Multiple choice answer set to choose the 1 correct answer
    return prob_answers, prob_operator, int_list_1, int_list_2, int_list_3, int_list_4, int_list_5, int_list_6, \
            integerDifferent_int_list_1, integerDifferent_int_list_2, integerDifferent_int_list_3, integerDifferent_int_list_4, \
            operatorDifferent_int_list_1, operatorDifferent_int_list_2, operatorDifferent_int_list_3, operatorDifferent_int_list_4, \
            show_center, interpret, mode, part, integerDifferent_marks, prob_images, prob_answer_set_images


def draw_combination(geom_conditions):
    img = 255 * np.ones((PANEL_SIZE, PANEL_SIZE), np.uint8)
    position = []
    condition_1 = geom_conditions[0]
    condition_2 = geom_conditions[1]
    geom_type = condition_1.type.get_value()
    geom_relation = condition_2.grelation.get_value()
    # print("In draw_combination, geom_relation: %s" % geom_relation) # For debugging purpose
    if geom_relation == "overlap":
        posit = draw_overlap(img, geom_type)
        position.extend(posit)
    elif geom_relation == "include":
        posit = draw_include(img, geom_type)
        position.extend(posit)
    elif geom_relation == "tangent":
        posit = draw_tangent(img, geom_type)
        position.extend(posit)
    return img, position


def draw_composition(geom_conditions):
    img = 255 * np.ones((PANEL_SIZE, PANEL_SIZE), np.uint8)
    position = []
    condition_1 = geom_conditions[0]
    condition_2 = geom_conditions[1]
    geom_type = condition_1.type.get_value()
    geom_format = condition_2.format.get_value()
    (x, y) = CENTER
    L = (LENGTH_2 * PANEL_SIZE + 2) * 1.4  # Size of the concrete circle shape
    L2 = 2 * LENGTH_2 * PANEL_SIZE + 4
    if geom_format == "line":
        posit = line_pos((x, y), L, L2)
    elif geom_format == "cross":
        posit = cross_pos((x, y), L, L2)
    elif geom_format == "triangle":
        posit = triangle_pos((x, y), L, L2)
    if geom_format == "square":
        posit = square_pos((x, y), L, L2)
    if geom_format == "circle":
        posit = circle_pos((x, y), L, L2 * 0.75)
    position.extend(posit)

    # Here is the actual concrete shape drawn
    if geom_type == "triangle":
        for c in posit:
            draw_triangle(img, c, L2 * 1.5, False, DEFAULT_WIDTH)
    elif geom_type == "square":
        for c in posit:
            draw_rectangle(img, c, L2, 0, DEFAULT_WIDTH)
    elif geom_type == "circle":
        for c in posit:
            draw_circle(img, c, L, DEFAULT_WIDTH)

    # self added
    elif geom_type == "hexagon":
        for c in posit:
            draw_hexagon(img, c, L, DEFAULT_WIDTH)
    elif geom_type == "rectangle":
        for c in posit:
            draw_rectangle(img, c, L * 0.95, 1, DEFAULT_WIDTH)
    return img, position


def draw_partition(geom_conditions):
    img = 255 * np.ones((PANEL_SIZE, PANEL_SIZE), np.uint8)
    position = []
    condition_1 = geom_conditions[0]
    condition_2 = geom_conditions[1]
    geom_type = condition_1.type.get_value()
    geom_part = condition_2.part.get_value()
    # print("geom_part: %s" % geom_part) # For debugging purpose

    L = LENGTH_3 * PANEL_SIZE
    L2 = 2 * LENGTH_3 * PANEL_SIZE

    if geom_type == "square":
        position = partition_square(img, CENTER, L2, geom_part, DEFAULT_WIDTH)
    elif geom_type == "circle":
        position = partition_circle(img, CENTER, L, geom_part, DEFAULT_WIDTH)
    elif geom_type == "hexagon":
        position = partition_hexagon(img, CENTER, L, geom_part, DEFAULT_WIDTH)
    # TODO: circle, and square 2,4, 6,8, rectangle and triangle
    # self add
    elif geom_type == "rectangle":
        position = partition_rectangle(img, CENTER, L, geom_part, DEFAULT_WIDTH)
    elif geom_type == "triangle":
        position = partition_triangle(img, CENTER, L, geom_part, DEFAULT_WIDTH)
    return img, position


'''
Below are more detailed functions used to draw panels.
1) Functions draw_overlap(), draw_include(), draw_tangent() draw the panels with different geometrical relations in 
combination problems. 
2) Functions line_pos(), cross_pos(), triangle_pos() and so on compute the coordinates of geometrical shapes in 
different arrangement formats of composition problems. 
3) Functions  partition_square(),  partition_hexagon(), partition_circle() cut certain geometrical shapes into certain 
parts. Function partition_pos() computes the coordinates of integers in partitioned shapes.
4) Functions draw_triangle(), draw_circle(), draw_hexagon(), draw_rectangle() draw basic geometrical shapes by lines, 
and return the coordinates that can be used to place integers.
'''


def draw_overlap(img, geom_type):
    if geom_type == "triangle": # An upright triangle is overlapped with an inverse triangle.
        scale = 1.3
        pos_1, pos_2, pos_3 = draw_triangle(img, CENTER, LENGTH_1 * 2 * PANEL_SIZE * scale, False, DEFAULT_WIDTH)
        pos_4, pos_5, pos_6 = draw_triangle(img, CENTER, LENGTH_1 * 2 * PANEL_SIZE * scale, True, DEFAULT_WIDTH)
        x,y = pos_1
        pos_1 = (x + LENGTH_1 * PANEL_SIZE * 0.1, y)
        x,y = pos_2
        pos_2 = (x - LENGTH_1 * PANEL_SIZE * 0.1, y)
        x,y = pos_3
        pos_3 = (x, y - LENGTH_1 * PANEL_SIZE * 0.1)
        x,y = pos_4
        pos_4 = (x + LENGTH_1 * PANEL_SIZE * 0.1, y) # bottom left, beside bottom middle point
        x,y = pos_5
        pos_5 = (x, y + LENGTH_1 * PANEL_SIZE * 0.1) # middle top
        x,y = pos_6
        pos_6 = (x - LENGTH_1 * PANEL_SIZE * 0.1, y) # bottom right, beside bottom middle point
        posit = [pos_4, pos_1, pos_5, pos_2, pos_6, pos_3]
    if geom_type == "square":
        pos_1, pos_2, pos_3, pos_4 = draw_rectangle(img, CENTER_1_1, LENGTH_1 * 2 * PANEL_SIZE, 0, DEFAULT_WIDTH)
        pos_5, pos_6, pos_7, pos_8 = draw_rectangle(img, CENTER_1_2, LENGTH_1 * 2 * PANEL_SIZE, 0, DEFAULT_WIDTH)
        # posit = [pos_5, pos_3]
        x5, y5 = pos_5
        pos_5_up = (x5, y5 - LENGTH_1 * PANEL_SIZE * 0.5)
        pos_5_down = (x5, y5 + LENGTH_1 * PANEL_SIZE * 0.5)
        x3, y3 = pos_3
        pos_3_up = (x3, y3 - LENGTH_1 * PANEL_SIZE * 0.5)
        pos_3_down = (x3, y3 + LENGTH_1 * PANEL_SIZE * 0.5)
        posit = [pos_5_up, pos_3_up, pos_5_down, pos_3_down]
    if geom_type == "rectangle": # Two rectangles with different orientations are overlapped.
        pos_1, pos_2, pos_3, pos_4 = draw_rectangle(img, CENTER, LENGTH_1 * PANEL_SIZE, 1, DEFAULT_WIDTH)
        pos_5, pos_6, pos_7, pos_8 = draw_rectangle(img, CENTER, LENGTH_1 * PANEL_SIZE, 2, DEFAULT_WIDTH)
        posit = [pos_5, pos_2, pos_7, pos_4]
    if geom_type == "circle":
        pos_1, pos_2, pos_3, pos_4 = draw_circle(img, CENTER_1_1, LENGTH_1 * PANEL_SIZE, DEFAULT_WIDTH)
        pos_5, pos_6, pos_7, pos_8 = draw_circle(img, CENTER_1_2, LENGTH_1 * PANEL_SIZE, DEFAULT_WIDTH)
        # posit = [pos_5, pos_3]
        x5, y5 = pos_5
        pos_5_up = (x5 + LENGTH_1 * PANEL_SIZE * 0.1, y5 - LENGTH_1 * PANEL_SIZE * 0.5)
        pos_5_down = (x5 + LENGTH_1 * PANEL_SIZE * 0.1, y5 + LENGTH_1 * PANEL_SIZE * 0.5)
        x3, y3 = pos_3
        pos_3_up = (x3 - LENGTH_1 * PANEL_SIZE * 0.1, y3 - LENGTH_1 * PANEL_SIZE * 0.5)
        pos_3_down = (x3 - LENGTH_1 * PANEL_SIZE * 0.1, y3 + LENGTH_1 * PANEL_SIZE * 0.5)
        posit = [pos_5_up, pos_3_up, pos_5_down, pos_3_down]
    if geom_type == "hexagon":
        pos_1, pos_2, pos_3, pos_4 = draw_hexagon(img, CENTER_1_1, LENGTH_1 * PANEL_SIZE, DEFAULT_WIDTH)
        pos_5, pos_6, pos_7, pos_8 = draw_hexagon(img, CENTER_1_2, LENGTH_1 * PANEL_SIZE, DEFAULT_WIDTH)
        posit = [pos_5, pos_2, pos_3, pos_8]
    return posit


def draw_include(img, geom_type):
    if geom_type == "triangle":
        pos_1, pos_2, pos_3 = draw_triangle(img, CENTER, LENGTH_1 * 2 * PANEL_SIZE * 1.25, False, DEFAULT_WIDTH)
        pos_4, pos_5, pos_6 = draw_triangle(img, CENTER, LENGTH_1 * PANEL_SIZE * 0.5, False, DEFAULT_WIDTH)
        x, y = pos_4
        pos_4 = (x - LENGTH_1 * PANEL_SIZE * 0.25, y)
        x, y = pos_5
        pos_5 = (x + LENGTH_1 * PANEL_SIZE * 0.25, y)
        x, y = pos_6
        pos_6 = (x, y + LENGTH_1 * PANEL_SIZE * 0.25)
        
        posit = [pos_1, pos_2, pos_3, pos_4, pos_5, pos_6]
    if geom_type == "square":
        pos_1, pos_2, pos_3, pos_4 = draw_rectangle(img, CENTER, LENGTH_1 * 2 * PANEL_SIZE, 0, DEFAULT_WIDTH)
        pos_5, pos_6, pos_7, pos_8 = draw_rectangle(img, CENTER, LENGTH_1 * PANEL_SIZE / 1.25, 0, DEFAULT_WIDTH)
        x, y = pos_1
        pos_1 = (x + LENGTH_1 * PANEL_SIZE * 0.2, y)
        x, y = pos_2
        pos_2 = (x, y + LENGTH_1 * PANEL_SIZE * 0.2)
        x, y = pos_3
        pos_3 = (x - LENGTH_1 * PANEL_SIZE * 0.2, y)
        x, y = pos_4
        pos_4 = (x, y - LENGTH_1 * PANEL_SIZE * 0.2)
        x, y = pos_5
        pos_5 = (x - LENGTH_1 * PANEL_SIZE * 0.05, y)
        x, y = pos_6
        pos_6 = (x, y - LENGTH_1 * PANEL_SIZE * 0.05)
        x, y = pos_7
        pos_7 = (x + LENGTH_1 * PANEL_SIZE * 0.1, y)
        x, y = pos_8
        pos_8 = (x, y + LENGTH_1 * PANEL_SIZE * 0.05)
        posit = [pos_1, pos_2, pos_3, pos_4, pos_5, pos_6, pos_7, pos_8]
    if geom_type == "rectangle":
        pos_1, pos_2, pos_3, pos_4 = draw_rectangle(img, CENTER, LENGTH_1 * PANEL_SIZE * 1.1, 1, DEFAULT_WIDTH)
        pos_5, pos_6, pos_7, pos_8 = draw_rectangle(img, CENTER, LENGTH_1 * PANEL_SIZE / 2.25, 1, DEFAULT_WIDTH)
        x, y = pos_5
        pos_5 = (x - LENGTH_1 * PANEL_SIZE * 0.15, y)
        x, y = pos_6
        pos_6 = (x, y - LENGTH_1 * PANEL_SIZE * 0.05)
        x, y = pos_7
        pos_7 = (x + LENGTH_1 * PANEL_SIZE * 0.15, y)
        x, y = pos_8
        pos_8 = (x, y + LENGTH_1 * PANEL_SIZE * 0.05)
        posit = [pos_1, pos_2, pos_3, pos_4, pos_5, pos_6, pos_7, pos_8]
    if geom_type == "circle":
        pos_1, pos_2, pos_3, pos_4 = draw_circle(img, CENTER, LENGTH_1 * PANEL_SIZE, DEFAULT_WIDTH)
        pos_5, pos_6, pos_7, pos_8 = draw_circle(img, CENTER, LENGTH_1 * PANEL_SIZE / 2.75, DEFAULT_WIDTH)
        x, y = pos_1
        pos_1 = (x + LENGTH_1 * PANEL_SIZE * 0.2, y)
        x, y = pos_2
        pos_2 = (x, y + LENGTH_1 * PANEL_SIZE * 0.2)
        x, y = pos_3
        pos_3 = (x - LENGTH_1 * PANEL_SIZE * 0.2, y)
        x, y = pos_4
        pos_4 = (x, y - LENGTH_1 * PANEL_SIZE * 0.2)
        x, y = pos_5
        pos_5 = (x - LENGTH_1 * PANEL_SIZE * 0.05, y)
        x, y = pos_6
        pos_6 = (x, y - LENGTH_1 * PANEL_SIZE * 0.05)
        x, y = pos_7
        pos_7 = (x + LENGTH_1 * PANEL_SIZE * 0.05, y)
        x, y = pos_8
        pos_8 = (x, y + LENGTH_1 * PANEL_SIZE * 0.05)
        posit = [pos_1, pos_2, pos_3, pos_4, pos_5, pos_6, pos_7, pos_8]
    if geom_type == "hexagon":
        pos_1, pos_2, pos_3, pos_4 = draw_hexagon(img, CENTER, LENGTH_1 * PANEL_SIZE * 1.2, DEFAULT_WIDTH)
        pos_5, pos_6, pos_7, pos_8 = draw_hexagon(img, CENTER, LENGTH_1 * PANEL_SIZE / 2, DEFAULT_WIDTH)
        x, y = pos_5
        pos_5 = (x - LENGTH_1 * PANEL_SIZE * 0.1, y)
        x, y = pos_6
        pos_6 = (x + LENGTH_1 * PANEL_SIZE * 0.1, y)
        x, y = pos_7
        pos_7 = (x + LENGTH_1 * PANEL_SIZE * 0.1, y)
        x, y = pos_8
        pos_8 = (x - LENGTH_1 * PANEL_SIZE * 0.1, y)
        posit = [pos_1, pos_2, pos_3, pos_4, pos_5, pos_6, pos_7, pos_8]
    return posit


def draw_tangent(img, geom_type):
    (x, y) = CENTER
    L = LENGTH_1 * PANEL_SIZE
    L2 = 2*L/3
    if geom_type == "triangle":
        c1 = (int(x - L/2), int(y + math.sqrt(3)*L/6))
        c2 = (int(x), int(y - math.sqrt(3)*L/3))
        c3 = (int(x + L/2), int(y + math.sqrt(3)*L/6))
        draw_triangle(img, c1, L, True, DEFAULT_WIDTH)
        draw_triangle(img, c2, L, True, DEFAULT_WIDTH)
        draw_triangle(img, c3, L, True, DEFAULT_WIDTH)
        posit = [c1, c2, c3]
    if geom_type == "square":
        c1 = (int(x - L/2), int(y - L/2))
        c2 = (int(x + L/2), int(y - L/2))
        c3 = (int(x), int(y + L/2))
        draw_rectangle(img, c1, L, 0, DEFAULT_WIDTH)
        draw_rectangle(img, c2, L, 0, DEFAULT_WIDTH)
        draw_rectangle(img, c3, L, 0, DEFAULT_WIDTH)
        posit = [c1, c2, c3]
    if geom_type == "rectangle":
        c1 = (int(x - L2), int(y - L2/2))
        c2 = (int(x + L2), int(y - L2/2))
        c3 = (int(x), int(y + L2/2))
        draw_rectangle(img, c1, L2, 1, DEFAULT_WIDTH)
        draw_rectangle(img, c2, L2, 1, DEFAULT_WIDTH)
        draw_rectangle(img, c3, L2, 1, DEFAULT_WIDTH)
        posit = [c1, c2, c3]
    if geom_type == "circle":
        c1 = (int(x - L2), int(y + math.sqrt(3)*L2/3))
        c2 = (int(x), int(y - 2*math.sqrt(3)*L2/3))
        c3 = (int(x + L2), int(y + math.sqrt(3)*L2/3))
        draw_circle(img, c1, L2, DEFAULT_WIDTH)
        draw_circle(img, c2, L2, DEFAULT_WIDTH)
        draw_circle(img, c3, L2, DEFAULT_WIDTH)
        posit = [c1, c2, c3]
    if geom_type == "hexagon":
        L = L * 0.75
        c1 = (int(x - L), int(y))
        c2 = (int(x + L/2), int(y - math.sqrt(3)*L/2))
        c3 = (int(x + L/2), int(y + math.sqrt(3)*L/2))
        draw_hexagon(img, c1, L, DEFAULT_WIDTH)
        draw_hexagon(img, c2, L, DEFAULT_WIDTH)
        draw_hexagon(img, c3, L, DEFAULT_WIDTH)
        posit = [c1, c2, c3]
    return posit


def line_pos((x, y), L, L2):
    c1 = (int(x), int(y - 1.8 * L2 - L))
    c2 = (int(x), int(y + 1.8 * L2 + L))
    c3 = (int(x), int(y - 0.25 * L2 - L))
    c4 = (int(x), int(y + 0.25 * L2 + L))
    posit = [c1, c2, c3, c4]
    return posit


# Cross_pos value after making it more compact
def cross_pos((x, y), L, L2):
    c1 = (int(x - 1.9 * L2 - L), y)
    c2 = (x, int(y - 1.9 * L2 - L))
    c3 = (int(x + 1.9 * L2 + L), y)
    c4 = (x, int(y + 1.9 * L2 + L))
    c5 = (int(x - 0.4 * L2 - L), y)
    c6 = (x, int(y - 0.4 * L2 - L))
    c7 = (int(x + 0.4 * L2 + L), y)
    c8 = (x, int(y + 0.4 * L2 + L))
    posit = [c1, c2, c3, c4, c5, c6, c7, c8]
    return posit

def triangle_pos((x, y), L, L2):
    c1 = (int(x - 2 * L2), int(y + 2 * math.sqrt(3) * L2 / 3))
    c2 = (int(x - L2), int(y - math.sqrt(3) * L2 / 3))
    c3 = (int(x), int(y - 4 * math.sqrt(3) * L2 / 3))
    c4 = (int(x + L2), int(y - math.sqrt(3) * L2 / 3))
    c5 = (int(x + 2 * L2), int(y + 2 * math.sqrt(3) * L2 / 3))
    c6 = (int(x), int(y + 2 * math.sqrt(3) * L2 / 3))
    posit = [c1, c2, c3, c4, c5, c6]
    return posit


def square_pos((x, y), L, L2):
    c1 = (int(x - 2 * L2), int(y - 2 * L2))
    c2 = (x, int(y - 2 * L2))
    c3 = (int(x + 2 * L2), int(y - 2 * L2))
    c4 = (int(x + 2 * L2), y)
    c5 = (int(x + 2 * L2), int(y + 2 * L2))
    c6 = (x, int(y + 2 * L2))
    c7 = (int(x - 2 * L2), int(y + 2 * L2))
    c8 = (int(x - 2 * L2), y)
    posit = [c1, c2, c3, c4, c5, c6, c7, c8]
    return posit


def circle_pos((x, y), L, L2):
    c1 = (int(x - 2 * math.sqrt(2) * L2), int(y))
    c2 = (int(x - 2 * L2), int(y - 2 * L2))
    c3 = (int(x), int(y - 2 * math.sqrt(2) * L2))
    c4 = (int(x + 2 * L2), int(y - 2 * L2))
    c5 = (int(x + 2 * math.sqrt(2) * L2), int(y))
    c6 = (int(x + 2 * L2), int(y + 2 * L2))
    c7 = (int(x), int(y + 2 * math.sqrt(2) * L2))
    c8 = (int(x - 2 * L2), int(y + 2 * L2))
    posit = [c1, c2, c3, c4, c5, c6, c7, c8]
    return posit


def partition_square(img, (x, y), l, part, thickness):
    draw_rectangle(img, (x, y), l, 0, thickness)
    p1 = (int(x - l/2), int(y - l/2))
    p2 = (int(x + l/2), int(y + l/2))
    p3 = (int(x - l/2), int(y + l/2))
    p4 = (int(x + l/2), int(y - l/2))
    p5 = (int(x - l/2), int(y))
    p6 = (int(x + l/2), int(y))
    p7 = (int(x), int(y - l/2))
    p8 = (int(x), int(y + l/2))
    if part == 2:
        cv2.line(img, p5, p6, (0, 0, 0), thickness) # Horizontal line
        posit = partition_pos(2, (x, y), math.sqrt(2)*l/4)
    if part == 4:
        cv2.line(img, p1, p2, (0, 0, 0), thickness)
        cv2.line(img, p3, p4, (0, 0, 0), thickness)
        posit = partition_pos(4, (x, y), math.sqrt(2)*l/4)
    if part == 6:
        cv2.line(img, p1, p2, (0, 0, 0), thickness)
        cv2.line(img, p3, p4, (0, 0, 0), thickness) # Vertical line
        cv2.line(img, p5, p6, (0, 0, 0), thickness) # Horizontal line
        posit = partition_pos(6, (x, y), math.sqrt(2)*l/4)
        x, y = posit[1]
        posit[1] = (x, y * 1.5)
        x, y = posit[4]
        posit[4] = (x, y * 0.95)
    if part == 8:
        cv2.line(img, p1, p2, (0, 0, 0), thickness) 
        cv2.line(img, p3, p4, (0, 0, 0), thickness)
        cv2.line(img, p5, p6, (0, 0, 0), thickness)
        cv2.line(img, p7, p8, (0, 0, 0), thickness)
        posit = partition_pos(8, (x, y), math.sqrt(2)*l/4)
    return posit

def partition_rectangle(img, (x, y), l, part, thickness):
    draw_rectangle(img, (x, y), l, 1, thickness)
    p1 = (int(x - l), int(y - l/2))
    p2 = (int(x + l), int(y + l/2))
    p3 = (int(x - l), int(y + l/2))
    p4 = (int(x + l), int(y - l/2))
    p5 = (int(x - l), int(y))
    p6 = (int(x + l), int(y))
    p7 = (int(x), int(y - l/2))
    p8 = (int(x), int(y + l/2))
    if part == 2:
        cv2.line(img, p5, p6, (0, 0, 0), thickness) # Horizontal line
        posit = partition_pos(2, (x, y), math.sqrt(2)*l/5)
    if part == 4:
        cv2.line(img, p1, p2, (0, 0, 0), thickness) # Diagonal line
        cv2.line(img, p3, p4, (0, 0, 0), thickness) # Diagonal line
        posit = partition_pos(4, (x, y), math.sqrt(2)*l/4)
        x, y = posit[0]
        posit[0] = (x * 0.75, y)
        x, y = posit[2]
        posit[2] = (x * 1.25, y)
        x, y = posit[1]
        posit[1] = (x, y * 1.1)
        x, y = posit[3]
        posit[3] = (x, y * 0.99)
    if part == 6:
        cv2.line(img, p1, p2, (0, 0, 0), thickness)
        cv2.line(img, p3, p4, (0, 0, 0), thickness)
        cv2.line(img, p5, p6, (0, 0, 0), thickness) # Horizontal line
        posit = partition_pos(6, (x, y), math.sqrt(2)*l/4)
        x, y = posit[0]
        posit[0] = (x * 0.55, y * 1.1)
        x, y = posit[2]
        posit[2] = (x * 1.25, y * 1.1)
        x, y = posit[3]
        posit[3] = (x * 1.25, y * 0.99)
        x, y = posit[5]
        posit[5] = (x * 0.55, y * 0.99)
        x, y = posit[1]
        posit[1] = (x, y * 1.25)
        x, y = posit[4]
        posit[4] = (x, y * 0.95)
    if part == 8:
        cv2.line(img, p1, p2, (0, 0, 0), thickness)
        cv2.line(img, p3, p4, (0, 0, 0), thickness)
        cv2.line(img, p5, p6, (0, 0, 0), thickness)
        cv2.line(img, p7, p8, (0, 0, 0), thickness)
        posit = partition_pos(8, (x, y), math.sqrt(2)*l/4)
        x, y = posit[0]
        posit[0] = (x * 0.55, y)
        x, y = posit[3]
        posit[3] = (x * 1.25, y)
        x, y = posit[4]
        posit[4] = (x * 1.25, y)
        x, y = posit[7]
        posit[7] = (x * 0.55, y)
        x, y = posit[1]
        posit[1] = (x * 0.95, y * 1.1)
        x, y = posit[2]
        posit[2] = (x * 1.05, y * 1.1)
        x, y = posit[5]
        posit[5] = (x * 1.05, y * 0.99)
        x, y = posit[6]
        posit[6] = (x * 0.95, y * 0.99)
    return posit

def partition_hexagon(img, (x, y), l, part, thickness):
    draw_hexagon(img, (x, y), l, thickness)
    p1 = (int(x - l), int(y))
    p2 = (int(x + l), int(y))
    p3 = (int(x - l/2), int(y - math.sqrt(3)*l/2))
    p4 = (int(x + l/2), int(y + math.sqrt(3)*l/2))
    p5 = (int(x - l/2), int(y + math.sqrt(3)*l/2))
    p6 = (int(x + l/2), int(y - math.sqrt(3)*l/2))
    p7 = (int(x), int(y - l * 0.85))   # Vertical line points
    p8 = (int(x), int(y + l * 0.85))   # Vertical line points

    if part == 2:
        # if sampled value of part is 4 or 2, partition the hexagon into 2 parts
        cv2.line(img, p1, p2, (0, 0, 0), thickness) # Horizontal line
        posit = partition_pos(2, (x, y), l/2)
    if part == 4:
        cv2.line(img, p3, p4, (0, 0, 0), thickness) # Diagonal line
        cv2.line(img, p5, p6, (0, 0, 0), thickness) # Diagonal line
        posit = partition_pos(4, (x, y), l/2)
    if part == 6 or part == 8:
        # if sampled value of part is 8 or 6, partition the hexagon into 6 parts
        cv2.line(img, p1, p2, (0, 0, 0), thickness)
        cv2.line(img, p3, p4, (0, 0, 0), thickness)
        cv2.line(img, p5, p6, (0, 0, 0), thickness)
        posit = partition_pos(6, (x, y), l/2)
    if part == 8:
        cv2.line(img, p1, p2, (0, 0, 0), thickness)
        cv2.line(img, p3, p4, (0, 0, 0), thickness)
        cv2.line(img, p5, p6, (0, 0, 0), thickness)
        cv2.line(img, p7, p8, (0, 0, 0), thickness)
        posit = partition_pos(8, (x, y), l/2)
    return posit


def partition_circle(img, (x, y), l, part, thickness):
    draw_circle(img, (x, y), l, thickness)
    p1 = (int(x - math.sqrt(2)*l/2), int(y - math.sqrt(2)*l/2))
    p2 = (int(x + math.sqrt(2)*l/2), int(y + math.sqrt(2)*l/2))
    p3 = (int(x - math.sqrt(2)*l/2), int(y + math.sqrt(2)*l/2))
    p4 = (int(x + math.sqrt(2)*l/2), int(y - math.sqrt(2)*l/2))
    p5 = (int(x - l), int(y))
    p6 = (int(x + l), int(y))
    p7 = (int(x), int(y - l))
    p8 = (int(x), int(y + l))
    p9 = (int(x - l/2), int(y - math.sqrt(3)*l/2))
    p10 = (int(x + l/2), int(y + math.sqrt(3)*l/2))
    p11 = (int(x - l/2), int(y + math.sqrt(3)*l/2))
    p12 = (int(x + l/2), int(y - math.sqrt(3)*l/2))
    if part == 2:
        cv2.line(img, p5, p6, (0, 0, 0), thickness)
        posit = partition_pos(2, (x, y), l/2)
    if part == 4:
        cv2.line(img, p1, p2, (0, 0, 0), thickness)
        cv2.line(img, p3, p4, (0, 0, 0), thickness)
        posit = partition_pos(4, (x, y), l/2)
    if part == 6:
        cv2.line(img, p5, p6, (0, 0, 0), thickness)
        cv2.line(img, p9, p10, (0, 0, 0), thickness)
        cv2.line(img, p11, p12, (0, 0, 0), thickness)
        posit = partition_pos(6, (x, y), l/2)
    if part == 8:
        cv2.line(img, p1, p2, (0, 0, 0), thickness)
        cv2.line(img, p3, p4, (0, 0, 0), thickness)
        cv2.line(img, p5, p6, (0, 0, 0), thickness)
        cv2.line(img, p7, p8, (0, 0, 0), thickness)
        posit = partition_pos(8, (x, y), l/2)
    return posit

def partition_triangle(img, (x, y), l, part, thickness):
    draw_triangle(img, (x, y), l * 2.1, False, thickness)
    p1 = (int(x - l/2), int(y - math.sqrt(3)*l/6))
    p2 = (int(x + l), int(y + math.sqrt(3)*l/3))
    p3 = (int(x - l), int(y + math.sqrt(3)*l/3))
    p4 = (int(x + l/2), int(y - math.sqrt(3)*l/6))
    p5 = (int(x - l), int(y))
    p6 = (int(x + l), int(y))
    p7 = (int(x), int(y - l * 1.2))
    p8 = (int(x), int(y + l/1.7))

    # Calculate midpoint between p1 and p3
    midpoint_p1_p3 = ((p1[0] + p3[0]) / 2, (p1[1] + p3[1]) / 2)

    if part == 2:
        cv2.line(img, p7, p8, (0, 0, 0), thickness) # Vertical line
        posit = partition_pos(2, (x, y), l/2)
        x, y = posit[0]
        posit[0] = (x * 0.75, y * 2)
        x, y1 = posit[1]
        posit[1] = (x * 1.25, y * 2)
    if part == 4:
        cv2.line(img, p1, p2, (0, 0, 0), thickness)
        cv2.line(img, p3, p4, (0, 0, 0), thickness)
        posit = partition_pos(4, (x, y), l/2)
        x, y = posit[0]
        posit[0] = (x * 1.1, y)
        x, y = posit[2]
        posit[2] = (x * 0.95, y)
        x, y = posit[3]
        posit[3] = (x, y * 0.9)
    if part == 6:
        cv2.line(img, p7, p8, (0, 0, 0), thickness)
        cv2.line(img, p1, p2, (0, 0, 0), thickness)
        cv2.line(img, p3, p4, (0, 0, 0), thickness)
        posit = partition_pos(6, (x, y), l/2)
        x, y = posit[0]
        posit[0] = (x * 1.25, y * 1.4)
        x1, y1 = posit[1]
        posit[1] = (x1 * 0.84, y1 * 1.5)
        x2, y2 = posit[2]
        posit[2] = (x2 * 0.82, y1 * 1.5)
        x3, y3 = posit[3]
        posit[3] = (x3 * 0.93, y * 1.4)
        x4, y4 = posit[4]
        posit[4] = (x2 * 0.82, y4 * 0.85)
        x5, y5 = posit[5]
        posit[5] = (x1 * 0.84, y5 * 1.025)
    if part == 8:
        cv2.line(img, p7, p8, (0, 0, 0), thickness) # Vertical line
        # Calculate positions for horizontal lines to split into 4 levels
        y_level1 = y - int(math.sqrt(3) * l / 5)
        y_level2 = y - int(math.sqrt(3) * l / 40)
        y_level3 = y + int(math.sqrt(3) * l / 6)

        x_left = int(x - l * 0.45)
        x_right = int(x + l * 0.475)
        x_left2 = int(x - l * 0.65)
        x_right2 = int(x + l * 0.675)
        x_left3 = int(x - l * 0.85)
        x_right3 = int(x + l * 0.85)
        
        cv2.line(img, (x_left, y_level1), (x_right, y_level1), (0, 0, 0), thickness)
        cv2.line(img, (x_left2, y_level2), (x_right2, y_level2), (0, 0, 0), thickness)
        cv2.line(img, (x_left3, y_level3), (x_right3, y_level3), (0, 0, 0), thickness)
        

        # # Draw the extended intersection line
        # cv2.line(img, p11, p12, (0, 0, 0), thickness)
        posit = partition_pos(8, (x, y), l/2)
        x1, y1 = posit[1]
        posit[1] = (x1, y1 * 1.1)
        x2, y2 = posit[2]
        posit[2] = (x2, y1 * 1.1)
        x, y = posit[0]
        posit[0] = (x1, y * 1.075)
        x3, y3 = posit[3]
        posit[3] = (x2, y * 1.075)
        x4, y4 = posit[4]
        posit[4] = (x2, y4 * 0.95)
        x5, y5 = posit[5]
        posit[5] = (x2, y5 * 0.945)
        x6, y6 = posit[6]
        posit[6] = (x1, y4 * 0.95)
        x7, y7 = posit[7]
        posit[7] = (x1, y5 * 0.945)
    return posit

def partition_pos(part, (x, y), r):
    # Compute the integer positions after partition.
    # These positions are on a circle with te same center as the partitioned geometrical shape.
    posit = []
    if part == 2:
        pos_1 = (int(x), int(y - r))
        pos_2 = (int(x), int(y + r))
        posit = [pos_1, pos_2]
    elif part == 4:
        pos_1 = (int(x - r), int(y))
        pos_2 = (int(x), int(y - r))
        pos_3 = (int(x + r), int(y))
        pos_4 = (int(x), int(y + r))
        posit = [pos_1, pos_2, pos_3, pos_4]
    elif part == 6:
        r *= 1.3    # Scale factor to make 80 x 80 be in correct position
        pos_1 = (int(x - math.sqrt(3)*r/2), int(y - r/2))
        pos_2 = (int(x), int(y - r))
        pos_3 = (int(x + math.sqrt(3)*r/2), int(y - r/2))
        pos_4 = (int(x + math.sqrt(3)*r/2), int(y + r/2))
        pos_5 = (int(x), int(y + r))
        pos_6 = (int(x - math.sqrt(3)*r/2), int(y + r/2))
        posit = [pos_1, pos_2, pos_3, pos_4, pos_5, pos_6]
    elif part == 8:
        r *= 1.3    # Scale factor to make 80 x 80 be in correct position
        scale = 0.7 # Scale factor to make 80 x 80 be in correct position
        # x control the position along the x-axis (horizontal)
        # y control the position along the y-axis (vertical)
        pos_1 = (int(x - math.sqrt(3)*r/2), int(y - scale * r/2))
        pos_2 = (int(x - scale * r/2), int(y - math.sqrt(3)*r/2))
        pos_3 = (int(x + scale * r/2), int(y - math.sqrt(3)*r/2))
        pos_4 = (int(x + math.sqrt(3)*r/2), int(y - scale * r/2))
        pos_5 = (int(x + math.sqrt(3)*r/2), int(y + scale * r/2))
        pos_6 = (int(x + scale * r/2), int(y + math.sqrt(3)*r/2))
        pos_7 = (int(x - scale * r/2), int(y + math.sqrt(3)*r/2))
        pos_8 = (int(x - math.sqrt(3)*r/2), int(y + scale * r/2))
        posit = [pos_1, pos_2, pos_3, pos_4, pos_5, pos_6, pos_7, pos_8]
    return posit


def draw_triangle(img, (x, y), l, inverse, thickness):
    # Parameter "inverse" indicates the orientation of triangle.
    # Draw upward triangle if parameter "inverse" is False, and draw downward triangle if parameter "inverse" is True.
    if inverse:
        p1 = (int(x - l/2), int(y - math.sqrt(3)*l/6))
        p2 = (int(x + l/2), int(y - math.sqrt(3)*l/6))
        p3 = (int(x), int(y + math.sqrt(3)*l/3))
        pos_1 = (int(x - 3*l/8), int(y + math.sqrt(3)*l/8))
        pos_2 = (int(x), int(y - math.sqrt(3)*l/4))
        pos_3 = (int(x + 3*l/8), int(y + math.sqrt(3)*l/8))
    else:
        p1 = (int(x - l/2), int(y + math.sqrt(3)*l/6))
        p2 = (int(x + l/2), int(y + math.sqrt(3)*l/6))
        p3 = (int(x), int(y - math.sqrt(3)*l/3))
        pos_1 = (int(x - 3*l/8), int(y - math.sqrt(3)*l/8))
        pos_2 = (int(x + 3*l/8), int(y - math.sqrt(3)*l/8))
        pos_3 = (int(x), int(y + math.sqrt(3)*l/4))
    cv2.line(img, p1, p2, (0, 0, 0), thickness)
    cv2.line(img, p2, p3, (0, 0, 0), thickness)
    cv2.line(img, p1, p3, (0, 0, 0), thickness)
    return pos_1, pos_2, pos_3


def draw_rectangle(img, (x, y), l, shape, thickness):
    # Parameter "shape" == 0 indicates a square
    # Parameter "shape" == 1 or 2 indicates the orientation of rectangle
    if shape == 0:
        p1 = (int(x - l/2), int(y - l/2))
        p2 = (int(x + l/2), int(y + l/2))
        pos_1 = (int(x - 3*l/4), int(y))
        pos_2 = (int(x), int(y - 3*l/4))
        pos_3 = (int(x + 3*l/4), int(y))
        pos_4 = (int(x), int(y + 3*l/4))
    elif shape == 1:
        p1 = (int(x - l), int(y - l/2))
        p2 = (int(x + l), int(y + l/2))
        pos_1 = (int(x - 5*l/4), int(y))
        pos_2 = (int(x), int(y - 3*l/4))
        pos_3 = (int(x + 5*l/4), int(y))
        pos_4 = (int(x), int(y + 3*l/4))
    elif shape == 2:
        p1 = (int(x - l/2), int(y - l))
        p2 = (int(x + l/2), int(y + l))
        pos_1 = (int(x - 3*l/4), int(y))
        pos_2 = (int(x), int(y - 5*l/4))
        pos_3 = (int(x + 3*l/4), int(y))
        pos_4 = (int(x), int(y + 5*l/4))
    cv2.rectangle(img, p1, p2, (0, 0, 0), thickness)
    return pos_1, pos_2, pos_3, pos_4


def draw_circle(img, (x, y), l, thickness):
    center = (int(x), int(y))
    pos_1 = (int(x - 3*l/2), int(y))
    pos_2 = (int(x), int(y - 3*l/2))
    pos_3 = (int(x + 3*l/2), int(y))
    pos_4 = (int(x), int(y + 3*l/2))
    cv2.circle(img, center, int(l), (0, 0, 0), thickness)
    return pos_1, pos_2, pos_3, pos_4


def draw_hexagon(img, (x, y), l, thickness):
    p1 = (int(x - l/2), int(y - math.sqrt(3)*l/2))
    p2 = (int(x + l/2), int(y - math.sqrt(3)*l/2))
    p3 = (int(x + l), int(y))
    p4 = (int(x + l/2), int(y + math.sqrt(3)*l/2))
    p5 = (int(x - l/2), int(y + math.sqrt(3)*l/2))
    p6 = (int(x - l), int(y))
    pos_1 = (int(x - 9*l/8), int(y - 3*math.sqrt(3)*l/8))
    pos_2 = (int(x + 9*l/8), int(y - 3*math.sqrt(3)*l/8))
    pos_3 = (int(x + 9*l/8), int(y + 3*math.sqrt(3)*l/8))
    pos_4 = (int(x - 9*l/8), int(y + 3*math.sqrt(3)*l/8))
    cv2.line(img, p1, p2, (0, 0, 0), thickness)
    cv2.line(img, p2, p3, (0, 0, 0), thickness)
    cv2.line(img, p3, p4, (0, 0, 0), thickness)
    cv2.line(img, p4, p5, (0, 0, 0), thickness)
    cv2.line(img, p5, p6, (0, 0, 0), thickness)
    cv2.line(img, p6, p1, (0, 0, 0), thickness)
    return pos_1, pos_2, pos_3, pos_4

