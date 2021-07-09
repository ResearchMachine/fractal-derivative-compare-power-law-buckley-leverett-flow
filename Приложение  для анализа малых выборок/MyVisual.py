#Перезагрузка ядра Jupyter Notebook для корректной визуализации интерфейса.
from IPython.core.display import HTML
HTML("<script>Jupyter.notebook.kernel.restart()</script>")

#Импорт стандартных библиотек. 
import time
import threading
import functools #используется функция partial

#Импорт сторонних библиотек.
#Импорт библиотеки для интерфейса.  
import ipywidgets as widgets
#Импорт библиотек для анализа данных.
import numpy as np
import pandas as pd
import math
#Импорт библиотек для визуализации.
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


df = pd.DataFrame()

#----------------------------------------------------------------------------------------
#Visualization data tab--------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
# Размер надписей на графиках
PLOT_LABEL_FONT_SIZE = 20 

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 16
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
# Генерация цветовой схемы
# Возвращает список цветов
def getColors(n):
    COLORS = []
    cm = plt.cm.get_cmap('hsv', n)
    for i in np.arange(n):
        COLORS.append(cm(i))
    return COLORS


def dict_sort(my_dict):
    keys = []
    values = []
    my_dict = sorted(my_dict.items(), key=lambda x:x[1], reverse=True)
    for k, v in my_dict:
        keys.append(k)
        values.append(v)
    return (keys,values)



#Box for choosing independed variable of modeling
Variables_VisualTab = widgets.SelectMultiple(
    options=list(df.columns)
)


VariablesAndText_VisualTab = widgets.HBox([widgets.Label('Переменные:'), 
                                     Variables_VisualTab])

#Combobox for choosing method of modeling
GraphType_VisualTab =  widgets.Dropdown(
    options=['Гистограмма (одномерный график)', 'Диаграмма рассеяния (двумерный график)'],
    description='Тип графика:'
)

#Button of data frame upload
ButtonBuildGraph_VisualTab = widgets.Button(
    description='Вывод графика'
)

def scatter_plot(feature, target):    
    plt.figure(figsize=(16, 8))
    plt.scatter(
        df[feature],
        df[target],
        c='black'
    )
    plt.xlabel(feature, fontsize=PLOT_LABEL_FONT_SIZE)
    plt.ylabel(target, fontsize=PLOT_LABEL_FONT_SIZE)
    plt.show()

    
    
def hist_plot(target):    
    Xplot = target
    Xplot_type_count = pd.value_counts(df[Xplot].values, sort=True)
    Xplot_count_keys, Xplot_count_values = dict_sort(dict(Xplot_type_count))    
    TOP_Xplot = len(Xplot_count_keys)
    OBJECT_COUNT = len(Xplot_count_keys)
    plt.title('Переменная '+Xplot, fontsize=PLOT_LABEL_FONT_SIZE)
    plt.bar(np.arange(TOP_Xplot), Xplot_count_values, color=getColors(TOP_Xplot))
    plt.xticks(np.arange(TOP_Xplot), Xplot_count_keys, rotation=0, fontsize=PLOT_LABEL_FONT_SIZE)
    plt.yticks(fontsize=PLOT_LABEL_FONT_SIZE)
    plt.ylabel('Количество наблюдений', fontsize=PLOT_LABEL_FONT_SIZE)
    plt.show()  
    

def float_hist_plot(target): 
    data = df[target]

    twentyfifth, seventyfifth, ninetyfifth = np.percentile(df[target], [10, 50, 90])
    data[data<0]=float('nan')
#     data.dropna(subset=[target, ])


    # Colours for different percentiles
    perc_25_colour = 'gold'
    perc_50_colour = 'mediumaquamarine'
    perc_75_colour = 'deepskyblue'
    perc_95_colour = 'peachpuff'

    # Plot the Histogram from the random data
    fig, ax = plt.subplots(figsize=(10,8))

    '''
    counts  = numpy.ndarray of count of data ponts for each bin/column in the histogram
    bins    = numpy.ndarray of bin edge/range values
    patches = a list of Patch objects.
            each Patch object contains a Rectnagle object. 
            e.g. Rectangle(xy=(-2.51953, 0), width=0.501013, height=3, angle=0)
    '''
    counts, bins, patches = ax.hist(df[target], 
                                    facecolor=perc_50_colour, 
                                    edgecolor='gray', 
                                    bins=1+math.trunc(math.log2(len(df[target]))))


    # Set the ticks to be at the edges of the bins.
    ax.set_xticks(bins.round(2))
    plt.xticks(rotation=70)

    # Set the graph title and axes titles
    #plt.title('Distribution of randomly generated numbers', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.xlabel(target, fontsize=20)

    # Change the colors of bars at the edges

    for patch, leftside, rightside in zip(patches, bins[:-1], bins[1:]):
        if rightside < twentyfifth:
            patch.set_facecolor(perc_25_colour)
        elif leftside > ninetyfifth:
            patch.set_facecolor(perc_95_colour)
        elif leftside > seventyfifth:
            patch.set_facecolor(perc_75_colour)

    # Calculate bar centre to display the count of data points and %
    bin_x_centers = 0.5 * np.diff(bins) + bins[:-1]
    bin_y_centers = ax.get_yticks()[1] * 0.25

    # Display the the count of data points and % for each bar in histogram
    for i in range(len(bins)-1):
        bin_label = "{0:,.2f}%".format((counts[i]/counts.sum())*100)
        plt.text(bin_x_centers[i], bin_y_centers, bin_label, rotation=90, rotation_mode='anchor', fontsize=15)




    #create legend
    handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [perc_25_colour, 
                                                             perc_50_colour, 
                                                             perc_75_colour, 
                                                             perc_95_colour]]
    labels= ["0-10 Percentile",
             "10-50 Percentile", 
             "50-90 Percentile", 
             ">90 Percentile"]


    # Annotation for bar values
    ax.annotate('N='+str(len(df[target]))
                +'\n 10 Percentile='+str(round(twentyfifth,4))
                +'\n 50 Percentile='+str(round(seventyfifth,4))
                +'\n 90 Percentile='+str(round(ninetyfifth,4)),
                xy=(.75,.50), xycoords='figure fraction',
                horizontalalignment='center', verticalalignment='bottom',
                fontsize=20, bbox=dict(boxstyle="round", fc="white"),
                rotation=0)
    plt.legend(handles, labels, bbox_to_anchor=(0.5, 0., 0.80, 0.99), fontsize=18)

    # Display the graph
    plt.show()


   
  
    
def Graph_Click(b, rs_=widgets.Output(layout={'border': '1px solid black'})):  
    GraphType = GraphType_VisualTab.value
    Variables = Variables_VisualTab.value
    
    with rs_:
        if GraphType=='Диаграмма рассеяния (двумерный график)':
            scatter_plot(Variables[0], Variables[1]) 
        if GraphType=='Гистограмма (одномерный график)':
            if df[Variables[0]].dtype=='float64':
                float_hist_plot(Variables[0])
            else:
                hist_plot(Variables[0]) 
        


#Box in the visualization tab
Box_Visual = widgets.VBox(
    children=(VariablesAndText_VisualTab,
              GraphType_VisualTab, 
              ButtonBuildGraph_VisualTab)          
)



def main(out, df1):
    global df
    df=df1.copy()
    ButtonBuildGraph_VisualTab.on_click(functools.partial(Graph_Click, rs_=out))
    Variables_VisualTab.options = list(df.columns)
    return(Box_Visual)
