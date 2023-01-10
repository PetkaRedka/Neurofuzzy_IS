import numpy as np
import torch
from torch import nn
from torch import optim
from pandas import read_csv
from math import isnan

#  Класс ANFIS-нейросети
class ANFIS:

    def __init__(self, x_array):
        self.x_array = x_array
        

    # Функция принадлежности
    def membership_function(self, a, b, c, x):
        if x <= a:
            return 0
        
        elif a <= x < b:
            return ((x - a) / (b - a))
        
        elif x == c:
            return 1

        elif b <= x <= c:
            return ((c - x) / (c - b))

        elif x > c:
            return 0
        



    # Первый слой - фазификация
    def fuzzification(self):

        low_x = [self.membership_function(0, 0, 5, x) for x in self.x_array]
        ave_x =  [self.membership_function(0, 5, 10, x) for x in self.x_array]
        high_x =  [self.membership_function(5, 10, 10, x) for x in self.x_array]
        
        return low_x, ave_x, high_x
    

    # Второй слой - база правил
    def rule_base(self, low_x, ave_x, high_x):
        
        # ----------- ПРАВИЛА ПРИ 3-х ВХОДНЫХ ПАРАМЕТРАХ ------------- #
        if len(low_x) == 3:

            rule1 = max([
                min(low_x[0], low_x[1], low_x[2]),
                min(ave_x[0], low_x[1], low_x[2]),
                min(low_x[0], ave_x[1], low_x[2])
            ])
            rule2 = max([
                min(high_x[0], low_x[1], low_x[2]),
                min(ave_x[0], ave_x[1], low_x[2]),
                min(low_x[0], high_x[1], low_x[2]),
                min(ave_x[0], high_x[1], low_x[2]),
                min(low_x[0], low_x[1], ave_x[2]),
                min(low_x[0], ave_x[1], ave_x[2]),
                min(low_x[0], high_x[1], ave_x[2]),
                min(low_x[0], low_x[1], high_x[2]),
            ])
            rule3 = max([
                min(high_x[0], ave_x[1], low_x[2]),
                min(high_x[0], high_x[1], low_x[2]),
                min(ave_x[0], low_x[1], ave_x[2]),
                min(high_x[0], low_x[1], ave_x[2]),
                min(ave_x[0], ave_x[1], ave_x[2]),
                min(ave_x[0], high_x[1], ave_x[2]),
                min(ave_x[0], low_x[1], high_x[2]),
                min(low_x[0], ave_x[1], high_x[2]),
                min(ave_x[0], ave_x[1], high_x[2]),
                min(low_x[0], high_x[1], high_x[2]),
            ])
            rule4 = max([
                min(high_x[0], ave_x[1], ave_x[2]),
                min(high_x[0], high_x[1], ave_x[2]),
                min(high_x[0], low_x[1], high_x[2]),
                min(high_x[0], ave_x[1], high_x[2]),
                min(ave_x[0], high_x[1], high_x[2])
            ])
            rule5 = min(high_x[0], high_x[1], high_x[2])

        # ----------- ПРАВИЛА ПРИ 2-х ВХОДНЫХ ПАРАМЕТРАХ ------------- #
        elif len(low_x) == 2:
            
            rule1 = min(low_x[0], low_x[1])
            rule2 = max([
                min(low_x[0], ave_x[1]), min(ave_x[0], low_x[1])
            ])
            rule3 = max([
                min(ave_x[0], ave_x[1]), min(high_x[0], low_x[1]),
                min(low_x[0], high_x[1])
            ])
            rule4 = max([
                min(ave_x[0], high_x[1]), min(high_x[0], ave_x[1])
            ])
            rule5 = min(high_x[0], high_x[1])

        return [rule1, rule2, rule3, rule4, rule5]


    # Третий слой - нормализация
    def normaliztion(self, rules):

      summ = sum(rules)  
      normal_w = []

      if summ == 0:
        return np.zeros(len(rules))


      for rule in rules:
        normal_w.append(rule / summ)
      
      return np.array(normal_w)


# Четвертый слой (вынесен отдельно) - дефаззификация
# Тут будут храниться обновленные веса, поэтому cлой вынесен отдельно
class Fourth_layer(nn.Module):

    def __init__(self, input_size, num_classes):

        super(Fourth_layer, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        return x


# Функция, обучющая нейросеть
def ANFIS_learning(X, Y, epochs_num, save_file):
    
    # Обозначим основные гиперпараметры
    lr = 0.01
    fourth_layer = Fourth_layer(X.shape[1], 5)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(fourth_layer.parameters(), lr=lr)
    L_train_acc = []
    loss = np.inf


    for epoch in range(epochs_num):

        L_acc = 0.
        answers = []

        for i in range(X.shape[0]):

            # ANFIS
            x = torch.from_numpy(np.array(X[i]))
            y = Y[i]
            
            # Инициализируем ANFIS (первые 3 слоя)
            anfis = ANFIS(x)
            # Фазифицируем
            low_x, ave_x, high_x = anfis.fuzzification()
            # Проходим через базу правил
            rules = anfis.rule_base(low_x, ave_x, high_x)
            # Нормализуем
            normal_w = anfis.normaliztion(rules)
            # Дефаззифицируем
            y_h = fourth_layer(x.float())
            # Складываем результат
            y_fin = torch.sum(y_h * torch.from_numpy(normal_w))
            
            # Считываем ошибку
            loss = criterion(y_fin, torch.tensor(y))

            if isnan(loss):
                answers.append(y)
                continue

            # Сумируем ошибку
            L_acc += loss.item()

            # Обратный проход с изменением весов
            # (Обратное распространение ошибки)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Сохраняем результаты текущей эпохи
            answers.append(y_fin.item())

        # Вычисляем среднюю ошибку эпохи
        L_train_acc.append(L_acc / X.shape[0])
        
    # Сохраняем модель после обучения
    torch.save(fourth_layer, save_file)
    return np.array(answers), L_train_acc



def ANFIS_get_result(X1, X2, X3):
    
    # Загружаем веса
    try:
        fourth_layer1 = torch.load("model1.pkl")
        fourth_layer2 = torch.load("model2.pkl")
        fourth_layer3 = torch.load("model3.pkl")
        fourth_layer4 = torch.load("model4.pkl")
    
    except BaseException:
        return None

    def one_layer_result(X, fourth_layer):

        # Проходим всю сеть и получаем результат
        x = torch.from_numpy(np.array(X))
        # Инициализируем ANFIS (первые 3 слоя)
        anfis = ANFIS(x)
        # Фазифицируем
        low_x, ave_x, high_x = anfis.fuzzification()
        # Проходим через базу правил
        rules = anfis.rule_base(low_x, ave_x, high_x)
        # Нормализуем
        normal_w = anfis.normaliztion(rules)
        # Дефаззифицируем
        y_h = fourth_layer(x.float())
        # Складываем результат
        y_fin = torch.sum(y_h * torch.from_numpy(normal_w))

        return y_fin.item()
    
    y1 = one_layer_result(X1, fourth_layer1)
    y2 = one_layer_result(X2, fourth_layer2)
    y3 = one_layer_result(X3, fourth_layer3)
    y4 = one_layer_result(np.array([y1, y2, y3]), fourth_layer4)

    return y4


# Проверка на тестовом датасете
def final_test():

    df = read_csv('learning_data.csv', sep=',', header=None)
    X1 = np.array(df.iloc[1000:, :3].values)
    X2 = np.array(df.iloc[1000:, 4:6].values)
    X3 = np.array(df.iloc[1000:, 7:9].values)
    Y = np.array(df.iloc[1000:, -1].values)
    
    criterion = nn.MSELoss()
    sum_loss = 0

    # Посчитаем среднюю ошибку на тестовых данных
    for i in range(X1.shape[0]):
        
        predicted_result = ANFIS_get_result(X1[i], X2[i], X3[i])
        sum_loss += criterion(torch.tensor(predicted_result), torch.tensor(Y[i]))
        
    # Финальная ошибка на тестовых данных
    return (sum_loss.item() / X1.shape[0])


# Функция для запуска обучения нейросети
def make_training(epoch_count):

    # Считаем наши тренировочные данные
    df = read_csv('learning_data.csv', sep=',', header=None)
    X1 = np.array(df.iloc[:1000, :3].values)
    Y1 = np.array(df.iloc[:1000, 3].values)
    X2 = np.array(df.iloc[:1000, 4:6].values)
    Y2 = np.array(df.iloc[:1000, 6].values)
    X3 = np.array(df.iloc[:1000, 7:9].values)
    Y3 = np.array(df.iloc[:1000, 9].values)
    Y4 = np.array(df.iloc[:1000, -1].values)

    Y1_predict, _ = ANFIS_learning(X1, Y1, epoch_count, "model1.pkl")
    Y2_predict, _ = ANFIS_learning(X2, Y2, epoch_count, "model2.pkl")
    Y3_predict, _ = ANFIS_learning(X3, Y3, epoch_count, "model3.pkl")

    X4 = np.array([[Y1_predict[i], Y2_predict[i], Y3_predict[i]] for i in range(len(Y1_predict))])
    _, loss = ANFIS_learning(X4, Y4, epoch_count, "model4.pkl")


    loss_after_tests = final_test()
    
    return loss, loss_after_tests