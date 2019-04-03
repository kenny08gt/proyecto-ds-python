#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 20:09:24 2019

@author: lionmane
"""
import numpy as np
#import seaborn as sns
#import pandas as pd
import matplotlib.pyplot as plt
data = np.load('proyecto_training_data.npy')
train_data_length = (int)(data.shape[0] * 0.8)
train_data = data[0:train_data_length,:]
test_data = data[train_data_length:,:]

sale_price_td = train_data[:,0]
overall_quality_td = train_data[:,1]
first_floor_square_feet_td = train_data[:,2]
total_rooms_td = train_data[:,3]
year_built_td = train_data[:,4]
loat_front_td = np.nan_to_num(train_data[:,5])

data_labels = ["Sales Price", "Overall material and finish quality", 'First Floor square feet', 'Total rooms above grade', 'Original construction date', 'Linear feet of street connected to property']
#
#def get_mean_max_min_range_dv(nd_array):
#    mean = np.mean(nd_array)
#    c_max = np.max(nd_array)
#    c_min = np.min(nd_array)
#    c_range = np.ptp(nd_array)
#    dv = np.std(nd_array)
#    return (mean, c_max, c_min, c_range, dv)
#
#def statistics_outputter(statistics, name):
#    print('Statistics of %s' % name)
#    print('Mean: %f' % statistics[0])
#    print('Max: %f' % statistics[1])
#    print('Mean: %f' % statistics[2])
#    print('Range: %f' % statistics[3])
#    print('Standar deviation: %f' % statistics[4])
#
def print_hr():
    print('***************************************************')
#    
#sale_price_statistics = get_mean_max_min_range_dv(sale_price_td)
#overall_quality_statistics = get_mean_max_min_range_dv(overall_quality_td)
#first_floor_square_feet_statistics = get_mean_max_min_range_dv(first_floor_square_feet_td)
#total_rooms_statistics = get_mean_max_min_range_dv(total_rooms_td)
#year_built_statisctics = get_mean_max_min_range_dv(year_built_td)
#loat_front_statistics = get_mean_max_min_range_dv(loat_front_td)
#
#print_hr()
#statistics_outputter(sale_price_statistics, data_labels[0])
#print_hr()
#statistics_outputter(overall_quality_statistics, data_labels[1])
#print_hr()
#statistics_outputter(first_floor_square_feet_statistics, data_labels[2])
#print_hr()
#statistics_outputter(total_rooms_statistics, data_labels[3])
#print_hr()
#statistics_outputter(year_built_statisctics, data_labels[4])
#print_hr()
#statistics_outputter(loat_front_statistics, data_labels[5])
#
#
##sns.distplot(sale_price_statistics)
##sns.distplot(overall_quality_statistics)
##sns.distplot(first_floor_square_feet_td, label = 'First Floor square feet')
##sns.distplot(total_rooms_td, label = 'Total rooms above grade')
##sns.distplot(year_built_td, label = 'Original construction date')
##sns.distplot(loat_front_td, label = 'Linear feet of street connected to property')
#
#def add_scatter_plot(x, y,x_label, y_label):
#    corr_coef = np.corrcoef(x, y)
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(1,1,1) 
#    
#    ax.scatter(x, y)
#    
#    ax.set_xlabel(x_label)
#    ax.set_ylabel(y_label)
#    ax.set_title('%s Vs %s (%f)' % (x_label, y_label, corr_coef[0,1]))
#
#    plt.show()    
#    
#print_hr()
#add_scatter_plot(overall_quality_td, sale_price_td, data_labels[1], data_labels[0])
#
#print_hr()
#add_scatter_plot(first_floor_square_feet_td, sale_price_td, data_labels[2], data_labels[0])
#
#print_hr()
#add_scatter_plot(total_rooms_td, sale_price_td, data_labels[3], data_labels[0])
#
#print_hr()
#add_scatter_plot(year_built_td, sale_price_td, data_labels[4], data_labels[0])
#
#print_hr()
#add_scatter_plot(loat_front_td, sale_price_td, data_labels[5], data_labels[0])
#
def train_model(x_nd_array, y_nd_array, epochs, imprimir_error_cada, learning_rate):

    print(y_nd_array.shape)
    mat = np.column_stack((x_nd_array, np.ones_like(x_nd_array)))
    print(mat.shape)
    parameter_arr = {}
    error_arr = []
    m = np.mean(y_nd_array) / np.mean(x_nd_array)
    b = y_nd_array - m*x_nd_array

    for i in range(epochs):
        vector = [m, np.mean(b)]
        parameter_arr[i] = vector
        y_hat = np.matmul(mat, vector)
        error = (0.5)*np.mean(np.power((y_hat[0] - y_nd_array) , 2))
        
        if i % imprimir_error_cada == 0:
            print(error)
        
        error_arr.append(error)
        
        gradiente_m = np.mean((y_hat[0] - y_nd_array)*x_nd_array)
        gradiente_b = np.mean(y_hat[1] - y_nd_array)
        
        m = m - learning_rate*gradiente_m
        b = b - learning_rate*gradiente_b
    
    return parameter_arr, error_arr

train_data_parameter_arr, train_data_error = train_model(overall_quality_td, sale_price_td, 10000, 10000, 0.0005)

#print(train_data_parameter_arr)
fig = plt.figure()
ax = fig.add_subplot(1,1,1) 


ax.grid()
ax.set_xlim(0, len(train_data_parameter_arr))
ax.set_ylim(np.min(train_data_error), np.max(train_data_error))
ax.plot(list(range(0, len(train_data_parameter_arr))), train_data_error)
ax.set_ylim(3e9,3.5e9)
ax.set_xlim(0,300)
ax.set_xlabel('# de iteración')
ax.set_ylabel('Error')
ax.set_title('Grafica de # de iteracion vrs error')

plt.show()
print_hr()

#print(train_data_parameter_arr[1][0])



def model_in_time(parameter_arr, n):
   
    
#    print(parameter_arr)
    plot_count = 1
    for i in range(len(parameter_arr)):    
        if i % n == 0:             
            fig = plt.figure()
            x_ax = []
            y_ax = []
            for j in range(12):
#                print(parameter_arr[i])
#                print('m %f en i %d' %(parameter_arr[i][0], i))
                x_ax.append(j)
                y_ax.append(parameter_arr[i][0]*j + parameter_arr[i][1])
            
            
            ax = fig.add_subplot(1, 1, 1)
            plt.plot(x_ax, y_ax, label = 'modelo en %d' % i)
            
            ax.set_xlim(0,12)
            ax.set_ylim(0,1000000)
            
            plt.scatter(overall_quality_td, sale_price_td, label = 'Overral Q vrs Sale Price',  color=[1, 0, 0],)
            
            ax.legend() #si no ejecutamos la función "legend" , no se mostraran los labels usados con plot
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.show()
        
#    ax.set_title('2 curvas en una gráfica')

 
    
#model_in_time(train_data_parameter_arr, 1000)
    
    
    
    
    