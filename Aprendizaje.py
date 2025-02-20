import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
import sys

from PyQt6.QtWidgets import QMainWindow, QApplication, QPushButton, QFileDialog, QLineEdit, QVBoxLayout, QWidget, QMessageBox, QListWidget, QHBoxLayout, QComboBox
from PyQt6.QtCore import pyqtSlot
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

class Main(QMainWindow):
    def __init__(self):
        super().__init__()

        self.end = False
        self.first_select = False
        self.regr_log = False
        self.regr_mult = False
        self.knn = False
        self.nameColumns = []
        self.nameRow = []
        self.cont_regr_mult = 0

        self.btn = QPushButton("Open file dialog", self)
        self.btn2 = QPushButton("Enter",self)
        self.btn3 = QPushButton("Next",self)
        self.ln = QLineEdit(self)
        self.list_widget_1 = QListWidget(self)
        self.list_widget_2 = QListWidget(self)

        self.cb = QComboBox(self)
        self.cb.addItems(["Regresión Lineal","Regresión Lineal Multiple","Regresión Logistica", "K-NN"])

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.btn)

        self.layout_h = QHBoxLayout()
        self.layout_h.addWidget(self.ln,75)
        self.layout_h.addWidget(self.btn2)
        self.layout_h.addWidget(self.cb)
        self.layout_h.addWidget(self.btn3)
        self.layout.addLayout(self.layout_h)

        self.h_layout = QHBoxLayout()
        self.h_layout.addWidget(self.list_widget_1)
        self.h_layout.addWidget(self.list_widget_2)
        self.layout.addLayout(self.h_layout)

        central_widget = QWidget()
        central_widget.setLayout(self.layout)

        self.setCentralWidget(central_widget)

        self.btn3.hide()

        self.btn.clicked.connect(self.open_dialog)
        self.btn2.clicked.connect(self.select_row)
        self.cb.currentIndexChanged.connect(self.change_cb)
        self.btn3.clicked.connect(self.next_row)


    @pyqtSlot()
    def change_cb(self):
        if self.cb.currentIndex() == 0:
            self.list_widget_1.clear()
            self.list_widget_1.addItem("Seleccione un archivo para realizar el analisis...")
            self.btn3.hide()
        if self.cb.currentIndex() == 1:
            self.list_widget_1.clear()
            self.list_widget_1.addItem("Seleccione la columna objetivo y presione NEXT...")
            self.btn3.show()
        if self.cb.currentIndex() == 2:
            self.list_widget_1.clear()
            self.list_widget_1.addItem("Seleccione las columnas de errores...")
            self.btn3.show()
        if self.cb.currentIndex() == 3:
            self.list_widget_1.clear()
            self.list_widget_1.addItem("Seleccione la columna objetivo...")
            self.btn3.show()


    @pyqtSlot()
    def open_dialog(self):
        fname = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "${HOME}",
            "CSV files (*.csv)",
        )

        self.df = pd.read_csv(fname[0])

        i = 0       
        while self.end == False:
            try:
                nombre = self.df.columns[i]
                i = i+1
            except Exception as e:
                self.end = True
            else:
                index = str(i-1)
                self.nameColumns.append(nombre)
                nombre = nombre + ' ' + index
                print(f'{nombre} [{i-1}]')
                self.list_widget_2.addItem(nombre)

        self.list_widget_1.addItem("Seleccione la primera columna de informacion (Eje X)...")
        
    @pyqtSlot()
    def select_row(self):
        if self.cb.currentIndex() == 0:
            self.analisisLineal()
        if self.cb.currentIndex() == 1:
            self.analisisLinealMultiple()
        if self.cb.currentIndex() == 2:
            self.regresionLogistica()
        if self.cb.currentIndex() == 3:
            self.Knn()   
        
    def analisisLineal(self):
        if self.end == True:
            lectura = int(self.ln.text())
            nombre = self.df.columns[lectura]
            if self.first_select == False:
                self.data_x = self.df[nombre]
                self.list_widget_1.addItem("Seleccione la segunda columna de informacion (Eje Y)...")
                self.first_select = True
            else:
                self.data_y = self.df[nombre]
                Xtrain, Xtest, Ytrain, Ytest = train_test_split(self.data_x, self.data_y, test_size=0.2)
                self.list_widget_1.addItem("El analisis se realizará en segundo plano...")
                #ANALISIS-------------------
                regr = LinearRegression()

                Xtrain = pd.DataFrame(Xtrain)
                Xtest = pd.DataFrame(Xtest)
                Ytrain = pd.DataFrame(Ytrain)
                Ytest = pd.DataFrame(Ytest)

                regr.fit(Xtrain, Ytrain)
                data_y_pred = regr.predict(Xtest)
                coef = "Coeficientes: {}".format(regr.coef_)
                error = "Error cuadratico medio: {:.2f}".format(mean_squared_error(Ytest, data_y_pred))
                r2 = "Coeficiente de determinación (1 = perfecto): {:.2f}".format(r2_score(Ytest,data_y_pred))
                self.list_widget_2.addItem("------------------------------------")
                self.list_widget_2.addItem(coef)
                self.list_widget_2.addItem(error)
                self.list_widget_2.addItem(r2)

                self.first_select = False

                plt.scatter(Xtest,Ytest,color="black")
                plt.plot(Xtest,data_y_pred,color="blue", linewidth=3)
                
                x_limits = plt.xlim()
                y_limits = plt.ylim()

                rango_x = x_limits[1] - x_limits[0]
                step_x = int(rango_x / 8)
                rango_y = y_limits[1] - y_limits[0]
                step_y = int(rango_y / 8)

                plt.xticks(np.arange(x_limits[0], x_limits[1], step=step_x))
                plt.yticks(np.arange(y_limits[0], y_limits[1], step=step_y))
                plt.show()
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("¡Advertencia!")
            msg.setText("Primero debe abrir un archivo en formato .csv")
            msg.exec()

    @pyqtSlot()
    def next_row(self):
        if self.cb.currentIndex() == 1:
            if self.regr_mult == False:   
                self.regr_mult = True
                select_row = int(self.ln.text())
                self.firstRow = self.df.columns[select_row]
            else:
                select_row = int(self.ln.text())
                self.nameRow.append(self.df.columns[select_row])
        if self.cb.currentIndex() == 2:
            select_row = int(self.ln.text())
            self.nameRow.append(self.df.columns[select_row])
            self.regr_log = True
        if self.cb.currentIndex() == 3:
            select_row = int(self.ln.text())
            self.nameRow.append(self.df.columns[select_row])
            self.knn = True

    def analisisLinealMultiple(self):
        if self.end == True and self.regr_mult == True:
            X = self.df[self.nameRow]
            Y = self.df[self.firstRow]
            Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)
            regr = LinearRegression()
            regr.fit(Xtrain, Ytrain)
            data_y_pred = regr.predict(Xtest)
            coef = "Coeficientes: {}".format(regr.coef_)
            error = "Error cuadratico medio: {:.2f}".format(mean_squared_error(Ytest, data_y_pred))
            r2 = "Coeficiente de determinación (1 = perfecto): {:.2f}".format(r2_score(Ytest,data_y_pred))
            self.list_widget_2.addItem("------------------------------------")
            self.list_widget_2.addItem(coef)
            self.list_widget_2.addItem(error)
            self.list_widget_2.addItem(r2)

            self.regr_mult = False
            
            plt.scatter(Ytest, data_y_pred, color="black", label="Predicciones")
            plt.plot([min(Ytest), max(Ytest)], [min(Ytest), max(Ytest)], color="blue", linestyle="--", label="Línea ideal")

            plt.xlabel("Valores reales")
            plt.ylabel("Predicciones")
            plt.title("Relación entre valores reales y predicciones")
            plt.legend()

            plt.show()
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("¡Advertencia!")
            msg.setText("Primero debe abrir un archivo en formato .csv")
            msg.exec()
    
    def regresionLogistica(self):
        if self.end == True and self.regr_log == True:
            Xrow_aux = [col for col in self.nameColumns if col not in self.nameRow]
            Xrow = []
            for col in Xrow_aux:
                try:
                    self.df[col] = self.df[col].astype(float)
                except Exception as e:
                    print(e)
                else:
                    Xrow.append(col)

            X = self.df[Xrow]
            
            self.list_widget_1.addItem("El analisis se realizará en segundo plano...")

            ventana = tk.Tk()
            ventana.geometry("1200x600")

            frame = tk.Frame(ventana)
            frame.pack(padx=10, pady=10)

            for i, col in enumerate(self.nameRow):
                Y = self.df[col]
                Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size=0.2)
                regr = LogisticRegression()
                regr.fit(Xtrain, Ytrain)
                y_pred = regr.predict(Xtest)
                
                row_idx = i // 3
                col_idx = i % 3

                subframe = tk.Frame(frame)
                subframe.grid(row = row_idx, column = col_idx, padx=10)

                label = tk.Label(subframe, text=col)
                label.pack()

                lista_datos1 = tk.Listbox(subframe)
                lista_datos1.pack(side = tk.LEFT, fill = tk.BOTH)

                lista_datos2 = tk.Listbox(subframe)
                lista_datos2.pack(side = tk.LEFT, fill = tk.BOTH)

                datos1 = Ytest
                datos2 = y_pred

                for j, dato in enumerate(datos1, start = 1):
                    color = "red" if dato == 0 else "green"
                    lista_datos1.insert(tk.END, f"{dato} ({j})")
                    lista_datos1.itemconfig(tk.END, bg=color)
                
                for j, dato in enumerate(datos2, start=1):
                    color = "red" if dato == 0 else "green"
                    lista_datos2.insert(tk.END, f"{dato} ({j})")
                    lista_datos2.itemconfig(tk.END, bg=color)

                accuracy = accuracy_score(Ytest, y_pred)
                accuracy_label = tk.Label(subframe, text=f"Accuracy: {accuracy:.2f}")
                accuracy_label.pack()

                precision = precision_score(Ytest, y_pred)
                precision_label = tk.Label(subframe, text=f"Precision: {precision:.2f}")
                precision_label.pack()

                conf_matrix = confusion_matrix(Ytest, y_pred)
                conf_matrix_str = "\n".join(" ".join(map(str, row)) for row in conf_matrix)
                conf_matrix_label = tk.Label(subframe, text=f"Matriz de confusión:\n{conf_matrix_str}")
                conf_matrix_label.pack()

            ventana.mainloop()
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("¡Advertencia!")
            msg.setText("Primero debe abrir un archivo en formato .csv")
            msg.exec()
    
    def Knn(self):
        if self.end == True and self.knn == True:
            Xrow_aux = [col for col in self.nameColumns if col not in self.nameRow]
            Xrow = []
            for col in Xrow_aux:
                try:
                    self.df[col] = self.df[col].astype(float)
                except Exception as e:
                    print(e)
                else:
                    Xrow.append(col)

            X = self.df[Xrow]
            
            self.list_widget_1.addItem("El analisis se realizará en segundo plano...")

            ventana = tk.Tk()
            ventana.geometry("1200x600")

            frame = tk.Frame(ventana)
            frame.pack(padx=10, pady=10)

            for i, col in enumerate(self.nameRow):
                Y = self.df[col]
                Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size=0.2)

                escalar = StandardScaler()
                Xtrain = escalar.fit_transform(Xtrain)
                Xtest = escalar.transform(Xtest)

                K_nn = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='minkowski', p=2)
                K_nn.fit(Xtrain,Ytrain)
                y_pred = K_nn.predict(Xtest)
                
                row_idx = i // 3
                col_idx = i % 3

                subframe = tk.Frame(frame)
                subframe.grid(row = row_idx, column = col_idx, padx=10)

                label = tk.Label(subframe, text=col)
                label.pack()

                lista_datos1 = tk.Listbox(subframe)
                lista_datos1.pack(side = tk.LEFT, fill = tk.BOTH)

                lista_datos2 = tk.Listbox(subframe)
                lista_datos2.pack(side = tk.LEFT, fill = tk.BOTH)

                datos1 = Ytest
                datos2 = y_pred

                for j, dato in enumerate(datos1, start = 1):
                    if dato == 0:
                        color = "red"
                    elif dato == 1:
                        color = "green"
                    elif dato == 2:
                        color = "blue"
                    elif dato == 3:
                        color = "yellow"
                    
                    lista_datos1.insert(tk.END, f"{dato} ({j})")
                    lista_datos1.itemconfig(tk.END, bg=color)
                
                for j, dato in enumerate(datos2, start=1):
                    if dato == 0:
                        color = "red"
                    elif dato == 1:
                        color = "green"
                    elif dato == 2:
                        color = "blue"
                    elif dato == 3:
                        color = "yellow"

                    lista_datos2.insert(tk.END, f"{dato} ({j})")
                    lista_datos2.itemconfig(tk.END, bg=color)

                accuracy = accuracy_score(Ytest, y_pred)
                accuracy_label = tk.Label(subframe, text=f"Accuracy: {accuracy:.2f}")
                accuracy_label.pack()

                precision = precision_score(Ytest, y_pred, average='micro')
                precision_label = tk.Label(subframe, text=f"Precision: {precision:.2f}")
                precision_label.pack()

                conf_matrix = confusion_matrix(Ytest, y_pred)
                conf_matrix_str = "\n".join(" ".join(map(str, row)) for row in conf_matrix)
                conf_matrix_label = tk.Label(subframe, text=f"Matriz de confusión:\n{conf_matrix_str}")
                conf_matrix_label.pack()

            ventana.mainloop()
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("¡Advertencia!")
            msg.setText("Primero debe abrir un archivo en formato .csv")
            msg.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_gui = Main()
    main_gui.show()
    sys.exit(app.exec())