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
# Тут будут храниться обновленные веса, поэтому для удобства
# Слой вынесен отдельно
class Fourth_layer(nn.Module):

    def __init__(self, input_size, num_classes):

        super(Fourth_layer, self).__init__()
        self.fc1 = nn.Linear(input_size, 5)
        
    def forward(self, x):
        x = self.fc1(x)
        return x


# Считаем наши тренировочные данные
df = read_csv('learning_data.csv', sep=',', header=None)
X = np.array(df.iloc[:, :-1].values)
Y = np.array(df.iloc[:, -1].values)

# Обозначим основные гиперпараметры
epochs_num = 5
lr = 0.01
fourth_layer = Fourth_layer(3, 5)
criterion = nn.MSELoss()
optimizer = optim.Adam(fourth_layer.parameters(), lr=lr)
L_train_acc = []
loss = np.inf


for epoch in range(epochs_num):

    L_acc = 0.
    
    for i in range(X.shape[0]):

        # ANFIS
        x = torch.from_numpy(np.array(X[i]))
        y = Y[i]
        
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
          continue

        # Сумируем ошибку
        L_acc += loss

        # Обратный проход с изменением весов
        # (Обратное распространение ошибки)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("prediction:", y_fin.item())
        print("true:", y)

    L_train_acc.append(L_acc / X.shape[0])

print(L_train_acc)