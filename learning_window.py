from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(400, 273)
        Form.setStyleSheet("QLabel\n"
"{\n"
"  font-weight:bold;\n"
"  font-size: 16px;\n"
"}\n"
"QPushButton:pressed{\n"
"    background-color: gray;\n"
"}"
"QPushButton\n"
"{\n"
"  border: 2px solid gray;\n"
"  border-radius: 10px;\n"
"  padding: 0 8px;\n"
"  background: white;\n"
"  font-size: 20px;\n"
"}")
        self.formLayout = QtWidgets.QFormLayout(Form)
        self.formLayout.setContentsMargins(30, 40, 30, -1)
        self.formLayout.setVerticalSpacing(10)
        self.formLayout.setObjectName("formLayout")
        self.epoch_lineEdit = QtWidgets.QLineEdit(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.epoch_lineEdit.sizePolicy().hasHeightForWidth())
        self.epoch_lineEdit.setSizePolicy(sizePolicy)
        self.epoch_lineEdit.setObjectName("epoch_lineEdit")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.epoch_lineEdit)
        self.epoch_label = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.epoch_label.sizePolicy().hasHeightForWidth())
        self.epoch_label.setSizePolicy(sizePolicy)
        self.epoch_label.setObjectName("epoch_label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.epoch_label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.learning_button = QtWidgets.QPushButton(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.learning_button.sizePolicy().hasHeightForWidth())
        self.learning_button.setSizePolicy(sizePolicy)
        self.learning_button.setObjectName("learning_button")
        self.horizontalLayout.addWidget(self.learning_button)
        self.formLayout.setLayout(1, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout)

        self.process_label = QtWidgets.QLabel(Form)
        sizePolicy.setHeightForWidth(self.process_label.sizePolicy().hasHeightForWidth())
        self.process_label.setSizePolicy(sizePolicy)
        self.process_label.setObjectName("epoch_label")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.process_label)

        self.empty_label = QtWidgets.QLabel(Form)
        sizePolicy.setHeightForWidth(self.empty_label.sizePolicy().hasHeightForWidth())
        self.empty_label.setSizePolicy(sizePolicy)
        self.empty_label.setObjectName("epoch_label")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.empty_label)

        self.back_button = QtWidgets.QPushButton(Form)
        self.back_button.setObjectName("back_button")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.back_button)
        
        self.graph_button = QtWidgets.QPushButton(Form)
        sizePolicy.setHeightForWidth(self.graph_button.sizePolicy().hasHeightForWidth())
        self.graph_button.setSizePolicy(sizePolicy)
        self.graph_button.setObjectName("graph_button")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.graph_button)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.epoch_label.setText(_translate("Form", "Количество эпох:"))
        self.process_label.setText(_translate("Form", "Статус: Неактивно"))
        self.empty_label.setText(_translate("Form", " "))
        self.learning_button.setText(_translate("Form", "Начать обучение"))
        self.back_button.setText(_translate("Form", "Назад"))
        self.graph_button.setText(_translate("Form", "Показать график ошибки"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
