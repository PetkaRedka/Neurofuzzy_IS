from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from main_window import Ui_mainWindow
from learning_window import Ui_Form
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from anfis import *
import numpy as np
import skfuzzy as fuzz


class MembershipGraph(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):

        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MembershipGraph, self).__init__(fig)


class GraphDialog(QDialog):
    def __init__(self, y_predicted, parent=None):
        super(GraphDialog, self).__init__(parent)

        sc = MembershipGraph(self, width=5, height=4, dpi=100)

        # Строим функции принадлежности и отмеаем наше значение
        y = np.arange(0, 10.1, 0.1)
        y_lo = fuzz.trimf(y, [0, 0, 2.5])  
        y_md = fuzz.trimf(y, [0, 2.5, 5])  
        y_ave = fuzz.trimf(y, [2.5, 5, 7.5])  
        y_dec = fuzz.trimf(y, [5, 7.5, 10])  
        y_gd = fuzz.trimf(y, [7.5, 10, 10])

        level_dict = {'Очень низкий:':y_lo, 
                'Низкий:': y_md, 
                'Средний:': y_ave, 
                'Высокий:': y_dec, 
                'Очень высокий:': y_gd}

        for key, value in level_dict.items():
            sc.axes.plot(y, value, label=key)

        sc.axes.annotate('Предсказанное значение', xy=(y_predicted, 0),  xycoords='data',
            xytext=(y_predicted, 0.5), textcoords='data',
            arrowprops=dict(facecolor='g'))
        
        sc.axes.legend()

        # Добавляем график на окно
        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout.addWidget(sc)
        self.setWindowTitle("График принадлежности")

        # Воспроизведем значения функций принадлежности
        mf_label = QLabel(self)
        mf_label.setObjectName("mf_label")
        self.verticalLayout.addWidget(mf_label)

        anfis = ANFIS([1, 1, 1])
        mf_values = []
        mf_values.append(anfis.membership_function(0, 0, 2.5, y_predicted))
        mf_values.append(anfis.membership_function(0, 2.5, 5, y_predicted))
        mf_values.append(anfis.membership_function(2.5, 5, 7.5, y_predicted))
        mf_values.append(anfis.membership_function(5, 7.5, 10, y_predicted))
        mf_values.append(anfis.membership_function(7.5, 10, 10, y_predicted))

        mf_label.setText('\n'.join((list(level_dict.keys())[i] + str(mf_values[i])) for i in range(len(mf_values))))


# Создадим класс основного окна
class Main(QMainWindow):

    def __init__(self, parent=None):

        QWidget.__init__(self, parent)
        self.ui = Ui_mainWindow()
        self.ui.setupUi(self)

        # Сразу добавим второй виджет
        self.Form = QWidget()
        self.learning_widget = Ui_Form()
        self.learning_widget.setupUi(self.Form)

        # Создадим стак окон
        self.ui.Stack.addWidget(self.Form)
      

        self.ui.start_button.clicked.connect(self.start_program)
        self.ui.graph_button.clicked.connect(self.show_fuzz_graph)
        self.ui.nn_button.clicked.connect(self.nn_settings)

    def start_program(self):
        
        x_arr = []
        
        # Проверяем корректность введенных данных 
        try:
            x1 = float(self.ui.x1_lineEdit.text())
            x2 = float(self.ui.x2_lineEdit.text())
            x3 = float(self.ui.x3_lineEdit.text())
            x4 = float(self.ui.x4_lineEdit.text())
            x5 = float(self.ui.x5_lineEdit.text())
            x6 = float(self.ui.x6_lineEdit.text())
            x7 = float(self.ui.x7_lineEdit.text())

            for elem in [x1, x2, x3, x4, x5, x6, x7]:
                if elem < 0: raise BaseException
                if elem > 10: raise BaseException

        except BaseException:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setText("Данные заполнены неправильно!")
            msgBox.setWindowTitle("Ошибка заполнения")
            msgBox.setStandardButtons(QMessageBox.Cancel)
            returnValue = msgBox.exec()
            return

        # Если все верно, тогда производим расчеты
        predicted_result = ANFIS_get_result(np.array([x1, x2, x3]), np.array([x4, x5]), np.array([x6, x7]))
        # Сообщение о том, что модель еще не обучена
        if predicted_result == None:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setText("Модель еще не обучена!")
            msgBox.setWindowTitle("Ошибка")
            msgBox.setStandardButtons(QMessageBox.Cancel)
            returnValue = msgBox.exec()
            return

        self.ui.y_lineEdit.setText(str(predicted_result))
        # Активируем кнопку "показать график"
        self.ui.graph_button.setEnabled(True)


    def show_fuzz_graph(self):

        dialog = GraphDialog(y_predicted=float(self.ui.y_lineEdit.text()))
        dialog.exec_()


    def back_button(self):
        # Переходим на основной экран
        self.ui.Stack.setCurrentWidget(self.ui.centralwidget)


    def nn_settings(self):
        
        # Переходим на новый экран
        self.ui.Stack.setCurrentWidget(self.Form)
        
        # Обозначим кнопку перехода в главное меню
        self.learning_widget.back_button.clicked.connect(self.back_button)


if __name__ == "__main__":

    import sys
    app = QApplication(sys.argv)
    myapp = Main()
    myapp.show()
    sys.exit(app.exec_())
