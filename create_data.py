import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from random import random
import csv

# Создает набор эталонный набор выходных данных
def compute_inputs(x1_input, x2_input, x3_input=None):

    x1 = ctrl.Antecedent(np.arange(0, 10.01, 0.01), 'x1')
    x2 = ctrl.Antecedent(np.arange(0, 10.01, 0.01), 'x2')
    x3 = ctrl.Antecedent(np.arange(0, 10.01, 0.01), 'x3')
    y1 = ctrl.Consequent(np.arange(0, 10.01, 0.01), 'y1')

    input_names = ['low', 'ave', 'high']
    output_names = ['low', 'med', 'ave', 'dec', 'high']
    x1.automf(names=input_names)
    x2.automf(names=input_names)
    x3.automf(names=input_names)
    y1.automf(names=output_names)

    # ------------------ БАЗА ПРАВИЛ ДЛЯ 3-х ВХОДОВ ----------------- #
    if x3_input:

        rule1 = ctrl.Rule((x1['low'] & x2['low'] & x3['low']) |
                        (x1['ave'] & x2['low'] & x3['low']) |
                        (x1['low'] & x2['ave'] & x3['low']), y1['low'])
        rule2 = ctrl.Rule((x1['high'] & x2['low'] & x3['low']) |
                        (x1['ave'] & x2['ave'] & x3['low']) |
                        (x1['low'] & x2['high'] & x3['low']) |
                        (x1['ave'] & x2['high'] & x3['low']) |
                        (x1['low'] & x2['low'] & x3['ave']) |
                        (x1['low'] & x2['ave'] & x3['ave']) |
                        (x1['low'] & x2['high'] & x3['ave']) |
                        (x1['low'] & x2['low'] & x3['high']), y1['med'])
        rule3 = ctrl.Rule((x1['high'] & x2['ave'] & x3['low']) |
                        (x1['high'] & x2['high'] & x3['low']) |
                        (x1['ave'] & x2['low'] & x3['ave']) |
                        (x1['high'] & x2['low'] & x3['ave']) |
                        (x1['ave'] & x2['ave'] & x3['ave']) |
                        (x1['ave'] & x2['high'] & x3['ave']) |
                        (x1['ave'] & x2['low'] & x3['high']) |
                        (x1['low'] & x2['ave'] & x3['high']) |
                        (x1['ave'] & x2['ave'] & x3['high']) |
                        (x1['low'] & x2['high'] & x3['high']), y1['ave'])
        rule4 = ctrl.Rule((x1['high'] & x2['ave'] & x3['ave']) |
                        (x1['high'] & x2['high'] & x3['ave']) |
                        (x1['high'] & x2['low'] & x3['high']) |
                        (x1['high'] & x2['ave'] & x3['high']) |
                        (x1['ave'] & x2['high'] & x3['high']), y1['dec'])
        rule5 = ctrl.Rule(x1['high'] & x2['high'] & x3['high'], y1['high'])
    
    # ------------------ БАЗА ПРАВИЛ ДЛЯ 2-х ВХОДОВ ----------------- #
    else:

        rule1 = ctrl.Rule(x1['low'] & x2['low'], y1['low'])
        rule2 = ctrl.Rule((x1['low'] & x2['ave']) | 
                          (x1['ave'] & x2['low']), y1['med'])
        rule3 = ctrl.Rule((x1['ave'] & x2['ave']) | 
                          (x1['high'] & x2['low']) |
                          (x1['low'] & x2['high']), y1['ave'])
        rule4 = ctrl.Rule((x1['ave'] & x2['high']) | 
                          (x1['high'] & x2['ave']), y1['dec'])
        rule5 = ctrl.Rule(x1['high'] & x2['high'], y1['high'])

    # Добавляем нашу базу данных для компьютеризации
    y1_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    y1_compute = ctrl.ControlSystemSimulation(y1_ctrl)

    # Выставялем начальные значения для подсчета
    y1_compute.input['x1'] = x1_input
    y1_compute.input['x2'] = x2_input
    if x3_input: y1_compute.input['x3'] = x3_input

    # Делаем расчет
    y1_compute.compute()
    # print(y1_compute.output['y1'])
    # y1.view(sim=y1_compute)
    return y1_compute.output['y1']


def create_some_noise(y4):
    return [round(random() / 5 + i, 3) for i in y4]

# Создает идеальные данные c шумами
def create_ideal(size):

    input_arr_y1 = [[round(random() * 10, 2) for j in range(3)] for i in range(size)]
    ideal_y1 = [round(compute_inputs(x1, x2, x3), 3) for x1, x2, x3 in input_arr_y1]

    input_arr_y2 = [[round(random() * 10, 2) for j in range(2)] for i in range(size)]
    ideal_y2 = [round(compute_inputs(x1, x2), 3) for x1, x2 in input_arr_y2]

    input_arr_y3 = [[round(random() * 10, 2) for j in range(2)] for i in range(size)]
    ideal_y3 = [round(compute_inputs(x1, x2), 3) for x1, x2 in input_arr_y3]

    ideal_y4 = [round(compute_inputs(ideal_y1[i], ideal_y2[i], ideal_y3[i]), 3) for i in range(size)]

    # Добавим немного шумов, чтобы данные не были такими идеальными
    y4 = create_some_noise(ideal_y4)
    
    # Теперь запишем CSV
    with open('learning_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(size):
            writer.writerow([input_arr_y1[i][0], input_arr_y1[i][1], input_arr_y1[i][2],
                            ideal_y1[i],
                            input_arr_y2[i][0], input_arr_y2[i][1],
                            ideal_y2[i],
                            input_arr_y3[i][0], input_arr_y3[i][1],
                            ideal_y3[i], 
                            y4[i]])

create_ideal(1300)
