import csv
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask,render_template,request
from collections import OrderedDict
import numpy as np
app = Flask(__name__, static_url_path='/static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.run(host = '0.0.0.0',port=80)

class BadData(Exception):
    pass


class AbstractCSVReader:
    def __init__(self, path):
        self.path = path

    def row_to_record(self, row):
        raise NotImplementedError

    def load(self):
        r_list = []
        with open(self.path, 'r') as in_file:
            reader = csv.DictReader(in_file)
            for row in reader:
                try:
                    r_list.append(self.row_to_record(row))
                except BadData:
                    print('BadData Exception:validation fails')
        in_file.close()
        return r_list


class GDPCSVReader(AbstractCSVReader):
    def row_to_record(self, row):
        if row['DATE'] and row['A191RL1Q225SBEA']:
            record = [row['DATE'], row['A191RL1Q225SBEA']]
        else:
            raise BadData
        return record


class INDPROCSVReader(AbstractCSVReader):
    def row_to_record(self, row):
        if row['DATE'] and row['INDPRO']:
            record = [row['DATE'], row['INDPRO']]
        else:
            raise BadData
        return record


class UNRATECSVReader(AbstractCSVReader):
    def row_to_record(self, row):
        if row['DATE'] and row['UNRATE']:
            record = [row['DATE'], row['UNRATE']]
        else:
            raise BadData
        return record


def newest(date1, date2, date3):
    if date3 > date1 and date2 > date1:
        return date1
    elif date1 > date2 and date3 > date2:
        return date2
    else:
        return date3


def oldest(date1, date2, date3):
    if date3 < date1 and date2 < date1:
        return date1
    elif date1 < date2 and date3 < date2:
        return date2
    else:
        return date3


@app.route('/', methods=['GET', 'POST'])
def init():
    #data load
    GDP_list = GDPCSVReader('./A191RL1Q225SBEA.CSV').load()
    INDPRO_list = INDPROCSVReader('./INDPRO.CSV').load()
    UNRATE_list = UNRATECSVReader('./UNRATE.CSV').load()
    #find time range
    start_year = oldest(GDP_list[0][0], INDPRO_list[0][0], UNRATE_list[0][0])[0:4]
    year_range = int(newest(GDP_list[len(GDP_list) - 1][0], INDPRO_list[len(INDPRO_list) - 1][0], UNRATE_list[len(UNRATE_list) - 1][0])[0:4]) - int(start_year)
    print(year_range)
    GDP_index, INDPRO_index, UNRATE_index = 0, 0, 0
    while GDP_list[GDP_index][0][0:4] != start_year:
        GDP_index += 4
    while INDPRO_list[INDPRO_index][0][0:4] != start_year:
        INDPRO_index += 12
    while UNRATE_list[UNRATE_index][0][0:4] != start_year:
        UNRATE_index += 12
    time_set = []
    y = []
    x = []
    INDPRO_value = []
    UNRATE_value = []
    #data format
    for i in range(year_range * 4):
        x_temp = []
        time_set.append(GDP_list[GDP_index][0][0:4])
        y.append(float(GDP_list[GDP_index][1]))
        GDP_index += 1
        x_temp.append((float(INDPRO_list[INDPRO_index][1]) + float(INDPRO_list[INDPRO_index + 1][1]) + float(INDPRO_list[INDPRO_index + 2][1])) / 3)
        INDPRO_value.append(x_temp[0])
        INDPRO_index += 3
        x_temp.append((float(UNRATE_list[UNRATE_index][1]) + float(UNRATE_list[UNRATE_index + 1][1]) + float(UNRATE_list[UNRATE_index + 2][1])) / 3)
        UNRATE_value.append(x_temp[1])
        UNRATE_index += 3
        x.append(x_temp)
    #store data
    np.save("x_storage", x)
    np.save("y_storage", y)
    np.save("time_storage", time_set)
    #draw picture
    plt.plot(time_set, INDPRO_value, color='navy', lw=2, label='INDPRO')
    plt.xlabel('date')
    plt.ylabel('INDPRO')
    plt.title('INDPRO')
    ax1 = plt.axes()
    ax1.xaxis.set_major_locator(plt.MaxNLocator(9))
    plt.legend()
    plt.savefig('static/INDPRO.png')
    plt.clf()
    plt.cla()
    plt.close()
    plt.plot(time_set, UNRATE_value, color='navy', lw=2, label='UNRATE')
    plt.xlabel('date')
    plt.ylabel('UNRATE')
    plt.title('UNRATE')
    ax2 = plt.axes()
    ax2.xaxis.set_major_locator(plt.MaxNLocator(9))
    plt.legend()
    plt.savefig('static/UNRATE.png')
    plt.close()
    return render_template('base.html', C_default = 1000, gamma_default = 0.4)

"""
    other model tried
    reg = LinearRegression().fit(x, y)
    print("linear")
    print(reg.score(x,y))
    print(reg.coef_)
    0.04774862972577265
    [-0.02663188  0.2692894 ]
    svr_lin = SVR(kernel='linear', C=1e3)
    print("svr_lin")
    svr_lin.fit(x,y)
    print(svr_lin.get_params())
    print(svr_lin.score(x,y))
    0.01976110377482998
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    print("svr_poly")
    svr_poly.fit(x,y)
    -2353.0260690695486
    print(svr_poly.get_params())
    print(svr_poly.score(x,y))"""
@app.route('/svr', methods=['GET', 'POST'])
def svr():
    if request.method == 'POST':
        #load data
        x = np.load("x_storage.npy")
        y = np.load("y_storage.npy")
        time_set = np.load("time_storage.npy")
        #regression
        svr_rbf = SVR(kernel='rbf', C=float(request.form['C']), gamma=float(request.form['gamma']))
        svr_rbf.fit(x,y)
        y_rbf = svr_rbf.predict(x)
        #draw picture
        plt.scatter(time_set, y, color='darkorange', label='data')
        plt.plot(time_set, y_rbf, color='navy', lw=2, label='RBF model')
        plt.xlabel('date')
        plt.ylabel('GDP')
        plt.title('Support Vector Regression')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        ax = plt.axes()
        ax.xaxis.set_major_locator(plt.MaxNLocator(9))
        plt.savefig('static/result.png')
        plt.close()
        return render_template('result.html', url='static/result.png', co=svr_rbf.score(x,y), pa=svr_rbf.get_params(), C_default = request.form['C'], gamma_default = request.form['gamma'])
