#Перезагрузка ядра Jupyter Notebook для корректной визуализации интерфейса.
from IPython.core.display import HTML
HTML("<script>Jupyter.notebook.kernel.restart()</script>")

#Импорт стандартных библиотек. 
import time
import threading
import functools #используется функция partial

#Импорт сторонних библиотек. 
#Импорт библиотек для анализа данных.
from math import isnan
import numpy as np
import pandas as pd
import math
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.svm import SVC
#Импорт библиотек для визуализации.
import numpy as np
import pandas as pd
from math import isnan


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import plot_confusion_matrix
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster

#standard libraries
import time
import threading
import functools
import importlib
import warnings


#visualisation libraries
import matplotlib.pyplot as plt
from ipywidgets import Layout
import ipywidgets as widgets

warnings.filterwarnings("ignore")




df = pd.DataFrame()
dfFloat = df.copy().loc[:, df.dtypes == float]
dfStrings = df.copy().loc[:, df.dtypes != float]


#---------------------------------------
#--------------------------Interface elements-------
#---------------------------------------

#----------------------------------------------------------------------------------------
#Linear Regression model accordion--------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#Combobox for choosing depended variable of modeling



DependVariable_MainAlgo_ModelingTab =  widgets.Dropdown(
    options=list(dfFloat.columns),
    description='Выберите зависимую переменную:',
    style={'description_width': 'initial'}
)


#Box for choosing independed variable of modeling
ExcludeColumns_MainAlgo_ModelingTab = widgets.SelectMultiple(
    options=list(dfFloat.columns),
    style={'description_width': 'initial'}
)
ExcludeColumnsAndText_MainAlgo_ModelingTab = widgets.HBox([widgets.Label(
    'Выберите переменные исключаемые из модели:'), 
                                     ExcludeColumns_MainAlgo_ModelingTab ])

#Combobox for choosing method of modeling
ComboboxFillNan_MainAlgo_ModelingTab =  widgets.Dropdown(
    options=['Среднее арифметическое', 
             'Медиана', 
             'Максимум', 
             'Минимум']
)  
ComboboxFillNanAndText_MainAlgo_ModelingTab = widgets.HBox([widgets.Label(
    'Заполнение пустых числовых ячеек в независимых переменных:'), 
                                     ComboboxFillNan_MainAlgo_ModelingTab])

MaxCorValue_MainAlgo_ModelingTab = widgets.BoundedFloatText(
    value=0.4,
    min=0.01,
    max=0.99,
    step=0.01
)
ComboboxMaxCorValueAndText_MainAlgo_ModelingTab = widgets.HBox([widgets.Label(
    'Минимальное значение коэффициента Спирмена для факторов:'), 
                                     MaxCorValue_MainAlgo_ModelingTab ])

MaxPValue_MainAlgo_ModelingTab = widgets.BoundedFloatText(
    value=0.1,
    min=0.001,
    max=0.2,
    step=0.001
)
ComboboxMaxPValueAndText_MainAlgo_ModelingTab = widgets.HBox([widgets.Label(
    'Минимальное значение p-value для факторов:'), 
                                     MaxPValue_MainAlgo_ModelingTab])

MaxVifValue_MainAlgo_ModelingTab = widgets.BoundedFloatText(
    value=10,
    min=6,
    max=12,
    step=0.001
)
ComboboxMaxVifValueAndText_MainAlgo_ModelingTab = widgets.HBox([widgets.Label(
    'Максимальное значение коэффициентов VIF для факторов:'), 
                                     MaxVifValue_MainAlgo_ModelingTab])

#Button of data frame upload
ButtonStart_MainAlgo_ModelingTab = widgets.Button(
    description='Выполнить'
)

#Box in the prediction tab
BoxMainAlgo_ModelingTab = widgets.VBox(
    children=(
        DependVariable_MainAlgo_ModelingTab,
        ExcludeColumnsAndText_MainAlgo_ModelingTab,
        ComboboxFillNanAndText_MainAlgo_ModelingTab,
        ComboboxMaxCorValueAndText_MainAlgo_ModelingTab,
        ComboboxMaxPValueAndText_MainAlgo_ModelingTab,
        ComboboxMaxVifValueAndText_MainAlgo_ModelingTab,
        ButtonStart_MainAlgo_ModelingTab
             )
)

WithoutMainAglo_ModelingTab = widgets.Label(value='Для запуска сэмплирование проведите построение модели')

#Box in the prediction tab
BoxSampleMainAlgo_ModelingTab = widgets.VBox(
    children=(WithoutMainAglo_ModelingTab,
             )
)

#----------------------------------------------------------------------------------------
#Correlation accordion--------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

#Combobox for choosing method of modeling
ComboboxCorrMethods_ModelingTab =  widgets.Dropdown(
    options=['Спирмена', 'Пирсона',  'Кенделла']
)
ComboboxCorrMethodsAndText_Cluster_ModelingTab = widgets.HBox([widgets.Label('Коэффициент корреляции:'), 
                                     ComboboxCorrMethods_ModelingTab])

ButtonCorrelation_ModelingTab = widgets.Button(
    description='Выполнить',
    style={'description_width': 'initial'}
)

BoxCorrelation_ModelingTab = widgets.VBox(
    children=(ComboboxCorrMethodsAndText_Cluster_ModelingTab, 
              ButtonCorrelation_ModelingTab)
)

#----------------------------------------------------------------------------------------
#Cluster accordion--------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#Box for choosing independed variable of modeling
ComboboxClusterMethods_ModelingTab =  widgets.Dropdown(
    options=['single' , 'complete', 'average']
)
ComboboxClusterMethodsAndText_Cluster_ModelingTab = widgets.HBox([widgets.Label('Метод определения расстояния:'), 
                                     ComboboxClusterMethods_ModelingTab])


Variable_Cluster_ModelingTab = widgets.SelectMultiple(
    options=list(dfFloat.columns)
)
VariableAndText_Cluster_ModelingTab = widgets.HBox([widgets.Label('Независимые переменные:'), 
                                     Variable_Cluster_ModelingTab])

Button_Cluster_ModelingTab = widgets.Button(
    description='Выполнить',
    style={'description_width': 'initial'}
)

Box_Cluster_ModelingTab = widgets.VBox(
    children=(ComboboxClusterMethodsAndText_Cluster_ModelingTab, 
              VariableAndText_Cluster_ModelingTab, 
              Button_Cluster_ModelingTab)
)


#----------------------------------------------------------------------------------------
#Diskriminant accordion--------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#Combobox for choosing depended variable of modeling
IndependVariable_Diskr_ModelingTab =  widgets.Dropdown(
    options=list(dfStrings.columns),
    description='Зависимая переменная:',
    style={'description_width': 'initial'}
)


#Box for choosing independed variable of modeling
DependVariable_Diskr_ModelingTab = widgets.SelectMultiple(options=list(dfFloat.columns))
DependVariableAndText_Diskr_ModelingTab = widgets.HBox([widgets.Label('Независимые переменные:'), 
                                     DependVariable_Diskr_ModelingTab])

Button_Diskr_ModelingTab = widgets.Button(
    description='Выполнить',
    style={'description_width': 'initial'}
)

Box_Diskr_ModelingTab = widgets.VBox(
    children=(IndependVariable_Diskr_ModelingTab,
        DependVariableAndText_Diskr_ModelingTab,              
             Button_Diskr_ModelingTab)
)

#----------------------------------------------------------------------------------------
#Accordion--------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
Accordion_Regression_ModelingTab = widgets.Accordion(children=[BoxMainAlgo_ModelingTab,
                                                              BoxSampleMainAlgo_ModelingTab])
Accordion_Regression_ModelingTab.set_title(0, 'Настройки условий расчета')
Accordion_Regression_ModelingTab.set_title(1, 'Прогнозирование (сэмплирование)')


Accordion_ModelingTab = widgets.Accordion(children=[BoxCorrelation_ModelingTab,
                                                    Box_Cluster_ModelingTab,
                                                    Box_Diskr_ModelingTab,
                                                    Accordion_Regression_ModelingTab
                                                    ])


Accordion_ModelingTab.set_title(0, 'Корреляционный анализ')
Accordion_ModelingTab.set_title(1, 'Иерархический кластерный анализ')
Accordion_ModelingTab.set_title(2, 'Дискриминантный анализ')
Accordion_ModelingTab.set_title(3, 'Линейная многомерная регрессионная модель')

#====


def Correlation_Click(b, out=widgets.Output()):
    MethodRu_CC = ComboboxCorrMethods_ModelingTab.value
    if MethodRu_CC=='Пирсона':      
        Method_CC = 'pearson'            
    if MethodRu_CC=='Спирмена':
        Method_CC = 'spearman'
    if MethodRu_CC=='Кенделла':
        Method_CC =  'kendall'       
    df_CC = df.copy()
    dfNames_CC = dict()
    for i in range(0, len(list(df))):
        dfNames_CC[list(df)[i]] = str(i)    

    df_CC = df_CC.rename(columns=dfNames_CC)

    fig, ax = plt.subplots(figsize=(25,25))
    sns.heatmap(df_CC.corr(method=Method_CC),
                               cbar=True,
                               annot=True, 
                               cmap="PiYG",
                               ax=ax) 
    
    with out:
        print('Корреляционная матрица, метод: '+MethodRu_CC+'\n')
        plt.show()
                
        
def Diskr_Modeling_Click(b, out=widgets.Output()):
    # Тренировка модели линейного дискриминантного анализа
    Target = IndependVariable_Diskr_ModelingTab.value
    Iwant = list(DependVariable_Diskr_ModelingTab.value)
    y = df[Target].copy()
    X = df[Iwant].copy().fillna(df[Iwant].mean())

    # Создание и тренировка объекта алгоритма линейного дискриминантного анализа
    clf = LinearDiscriminantAnalysis()
    clf_fitted = clf.fit(X, y)
    plot_confusion_matrix(clf_fitted, X, y, cmap=plt.cm.Blues) 
    
    with out:
        print('Матрица ошибок')
        plt.show()

        
def Cluster_Modeling_Click(b, out=widgets.Output()):
    wantColumns = list(Variable_Cluster_ModelingTab.value)
    KlasterMethod = ComboboxClusterMethods_ModelingTab.value    
    KlasterData = df.loc[0:, wantColumns]
    KlasterData = KlasterData.dropna(subset=wantColumns)
    
    # Исключаем информацию об образцах, сохраняем для дальнейшего использования
    varieties = list(range(1, len(df.loc[0:, wantColumns[0]])+1)) 

    scaler = MinMaxScaler()
    scaler.fit(KlasterData)
    NormKlasterData =pd.DataFrame(scaler.transform(KlasterData), columns = KlasterData.columns)

    # Извлекаем измерения как массив NumPy
    samples = NormKlasterData.values

    # # Реализация иерархической кластеризации при помощи функции linkage
    mergings = linkage(samples, method=KlasterMethod, metric = 'euclidean')
    nodes = fcluster(mergings, 2, criterion="maxclust")

    # Строим дендрограмму, указав параметры удобные для отображения
    fig, ax = plt.subplots(figsize=(12,5)) 
    dendrogram(mergings,
               labels=varieties,
               leaf_rotation=90,
               leaf_font_size=15,
               ax=ax)

    with out:
        plt.show()
        print('\n--------------------------------------------------------- ')
        print('Метрика силуэта (качества, 1 - идеальная кластеризация): ')
        print(float(silhouette_score(samples , nodes, metric='euclidean')))
        print('---------------------------------------------------------')
        
def CorrelSelection(Variable, DataFrame, ExcludeColumns, MinCorrValue):  
    df_CC = DataFrame.copy()
    CorrValues = {}

    #удалить стрингпеременные    
    dfFloatColumns = DataFrame.loc[:, DataFrame.dtypes == float]
    OtherVariables = list(dfFloatColumns.columns)

        
    for i in OtherVariables:
            CorrValues[i] = df_CC[Variable].corr(df_CC[i], method='spearman')
            
    clean_dict = {k: CorrValues[k] for k in CorrValues if not isnan(CorrValues[k])}       
    list_d = list(clean_dict.items())
    list_d.sort(key=lambda i: abs(i[1]))
    
    TopColumns = []
 
    for i in reversed(list_d):
        key=0
        for k in ExcludeColumns:
            if i[0]==k:
                key=1        
        if abs(i[1])>MinCorrValue and key==0:
            TopColumns.append(i[0])
    return(TopColumns)


def PvalueSelection(Variable, DataFrame, HowFill, WantColumns, MaxPValue): 
    df_PS = DataFrame.copy()
    if WantColumns!=[]:
        dataForAnalysis = df_PS.dropna(subset=[Variable, ])
        yForAnalysis = dataForAnalysis.loc[:, Variable]
        if HowFill=='Среднее арифметическое':
            dataForAnalysis = dataForAnalysis.fillna(dataForAnalysis.mean())
        if HowFill=='Медиана':
            dataForAnalysis = dataForAnalysis.fillna(dataForAnalysis.median())  
        if HowFill=='Максимум':
             dataForAnalysis = dataForAnalysis.fillna(dataForAnalysis.max())
        if HowFill=='Минимум':
             dataForAnalysis = dataForAnalysis.fillna(dataForAnalysis.min())      
        dataForAnalysis = dataForAnalysis.fillna(dataForAnalysis.mean())
        xForAnalysis = dataForAnalysis.loc[:, WantColumns]
        dataForAnalysis = dataForAnalysis[WantColumns]


        CurrentX = WantColumns.copy()
        maxPValue = 1
        while maxPValue>MaxPValue:
            xForAnalysis = dataForAnalysis.loc[:, CurrentX]
            regressor = sm.OLS(list(yForAnalysis), add_constant(xForAnalysis)).fit()
            PValues = regressor.pvalues
            maxPValue=max(PValues)

            if maxPValue>MaxPValue:
                for i in CurrentX:
                    if PValues[i]==maxPValue:
                        CurrentX.remove(i)
        return(CurrentX) 
    else:
        return(0)

    
def VifValueSelection(Variable, DataFrame, HowFill, WantColumns, MaxVifValue): 
    if HowFill=='Среднее арифметическое':
        df_VS = DataFrame.copy().fillna(DataFrame.mean())
    if HowFill=='Медиана':
        df_VS = DataFrame.copy().fillna(DataFrame.median())  
    if HowFill=='Максимум':
         df_VS = DataFrame.copy().fillna(DataFrame.max())
    if HowFill=='Минимум':
         df_VS = DataFrame.copy().fillna(DataFrame.min()) 
    
    xColumns = []   
    for i in range(0, len(WantColumns)):
        xColumns.append(WantColumns[i])
        xForAnalysis = df_VS.loc[:, xColumns]
        
        X = add_constant(xForAnalysis)
        VifValues = pd.Series([variance_inflation_factor(X.values, i) 
                       for i in range(X.shape[1])], 
                      index=X.columns).drop(labels=['const', ])
        maxVifValue=max(VifValues)
        if maxVifValue>MaxVifValue:
            xColumns.remove(WantColumns[i])       
    return(xColumns)   


def MainAlgoResults(Variable, DataFrame, HowFill, WantColumns, out):   
    # уравнение регрессии,
    df_MAR = DataFrame.copy()
    dataForAnalysis = df_MAR.dropna(subset=[Variable, ])
    yForAnalysis = dataForAnalysis.loc[:, Variable]
    if HowFill=='Среднее арифметическое':
        dataForAnalysis = dataForAnalysis.fillna(dataForAnalysis.mean())
    if HowFill=='Медиана':
        dataForAnalysis = dataForAnalysis.fillna(dataForAnalysis.median())  
    if HowFill=='Максимум':
         dataForAnalysis = dataForAnalysis.fillna(dataForAnalysis.max())
    if HowFill=='Минимум':
         dataForAnalysis = dataForAnalysis.fillna(dataForAnalysis.min())       
    xForAnalysis = dataForAnalysis.loc[:, WantColumns]
    dataForAnalysis = dataForAnalysis[WantColumns]    
    regressor = sm.OLS(list(yForAnalysis), add_constant(xForAnalysis)).fit()
    with out:
        print('\n--------------------------------------------------------- ')
        print('Коэффициенты регрессионного уравнения')
        print(regressor.params)
        print('--------------------------------------------------------- ')
    
    y1 = yForAnalysis
    y2 = regressor.predict(add_constant(xForAnalysis))
    
    x = list(range(1, len(y1)+1))
    
    
    fig, ax = plt.subplots(figsize=(7,4))
    #ax.scatter(x, y1)
    #ax.scatter(x, y2)
    ax.scatter(y1, y2)

    ax.set_xlabel(Variable+' (Факт)', fontsize=15)
    ax.set_ylabel('Прогноз', fontsize=15)
    with out:   
        plt.show()
    
    with out:
        print('\n--------------------------------------------------------- ')
        print('\nКоэффициент детерминации:')
        print(regressor.rsquared)
        print('\nСкорректированный коэффициент детерминации\n(даёт штраф за дополнительно включённые факторы):')
        print(regressor.rsquared_adj)
        print('--------------------------------------------------------- ')
   
    
    ax1 = plt.hist(regressor.resid)
    plt.xlim(-200,200)
    plt.xlabel('Остатки', fontsize=15)    
    with out:
        print('\nГистограмма остатков:')
        plt.show()


    with out:
        print('\n--------------------------------------------------------- ')
        print('\nCреднеквадратичное отклонение остатков:')
        print(np.std(regressor.resid))
        print('\nГраницы интервалов для коэффициентов регрессионного уравнения, \nв которых сосредоточено 10% возможных значений:')
        print(regressor.conf_int(alpha=0.9))
        print('--------------------------------------------------------- ')
    return(regressor.conf_int(alpha=0.9))

def VisualBorderSample(Name):   
    NameOfVar= widgets.Label(value=Name)
    LeftBound = widgets.FloatText(description='Левая')
    RightBound = widgets.FloatText(description='Правая')        
    return([NameOfVar, LeftBound, RightBound])

def MainAlgo_Modeling_Click(b, out=widgets.Output()):
    global SetOfTools, ListOfNames
    # Тренировка модели линейного дискриминантного анализа
    Target = DependVariable_MainAlgo_ModelingTab.value 
    ExcludeColumns = list(ExcludeColumns_MainAlgo_ModelingTab.value) 
    HowFill = ComboboxFillNan_MainAlgo_ModelingTab.value
    MaxCorValue = MaxCorValue_MainAlgo_ModelingTab.value
    MaxPValue = MaxCorValue_MainAlgo_ModelingTab.value
    MaxVifValue = MaxVifValue_MainAlgo_ModelingTab.value
    
    ColumnWithCorr = CorrelSelection(Target, 
                                     df,
                                     ExcludeColumns, 
                                     MaxCorValue)
    if ColumnWithCorr!=[]:
        with out: 
            print('\nПеременные прошедшие селекцию по показателю корреляции: \n')
            print(ColumnWithCorr)
        ColumnWithCorrAndPvalue = PvalueSelection(Target, 
                                                  df,
                                                  HowFill,
                                                  ColumnWithCorr, 
                                                  MaxPValue)
        with out:         
            print('\nПеременные прошедшие селекцию по P-value: \n')
            print(ColumnWithCorrAndPvalue)


        if ColumnWithCorrAndPvalue!=[]:          
            ColumnWithCorrAndPvalueAndVif = VifValueSelection(Target, 
                                                              df,
                                                              HowFill,
                                                              ColumnWithCorrAndPvalue, 
                                                              MaxVifValue)
            with out: 
                print('\nПеременные прошедшие селекцию по VIF критерию: \n')
                print(ColumnWithCorrAndPvalueAndVif)
            ConfIntervals = MainAlgoResults(Target, 
                                            df, 
                                            HowFill, 
                                            ColumnWithCorrAndPvalueAndVif, 
                                            out)

            ListOfNames = ['const']
            ListOfNames.extend(ColumnWithCorrAndPvalueAndVif)

            Header = widgets.Label(value='Границы сэмплирования для переменных: ')
            SetOfTools = [Header]
            for i in ListOfNames:
                SetOfTools.append(VisualBorderSample(i)[0])
                SetOfTools.append(VisualBorderSample(i)[1])
                SetOfTools.append(VisualBorderSample(i)[2])
                
            SetOfTools[2].value = ConfIntervals[0]['const']
            SetOfTools[3].value = ConfIntervals[1]['const']                
            for i in range(1, len(ListOfNames)):
                SetOfTools[2+3*i].value = df[ListOfNames[i]].mean()*0.9
                SetOfTools[3+3*i].value = df[ListOfNames[i]].mean()*1.1
            


            #Combobox for choosing method of modeling
            ComboboxHowSample_MainAlgo_ModelingTab =  widgets.Dropdown(
                options=['Нормальное', 
                         'Равномерное']
            )  
            ComboboxHowSampleAndText_MainAlgo_ModelingTab = widgets.HBox([widgets.Label(
                'Тип распределения:'), 
                                                 ComboboxHowSample_MainAlgo_ModelingTab])


            SetOfTools.append(ComboboxHowSampleAndText_MainAlgo_ModelingTab)

            ButtonStart_SampleMainAlgo_ModelingTab = widgets.Button(
                description='Выполнить'
            )
            ButtonStart_SampleMainAlgo_ModelingTab.on_click(functools.partial(Sample_Modeling_Click, out=out))

            

            #Combobox for choosing method of modeling
            HowMuch_SampleMainAlgo_ModelingTab =widgets.IntText(
                value=100000
            )

            HowMuchAndText_SampleMainAlgo_ModelingTab = widgets.HBox([widgets.Label('Объем выборки:'), 
                                                 HowMuch_SampleMainAlgo_ModelingTab])
            
            SetOfTools.append(HowMuchAndText_SampleMainAlgo_ModelingTab)
            SetOfTools.append(ButtonStart_SampleMainAlgo_ModelingTab)

            BoxSampleMainAlgo_ModelingTab.children = SetOfTools
            
        else:
            with out: 
                print('Факторов с заданным VIF параметров нет, алгоритм окончен, модель не построена')

                

#             print('Факторов с заданным уровнем корреляции нет, алгоритм окончен, модель не построена')
def Sample_Modeling_Click(b, out=widgets.Output()):
    wantColumns1 = ListOfNames.copy()
    DataFrame1 = df.copy()
    Variable1 = DependVariable_MainAlgo_ModelingTab.value 
    HowFill1 = ComboboxFillNan_MainAlgo_ModelingTab.value
    VolumeOfSample1 = SetOfTools[len(SetOfTools)-2].children[1].value
    out1 = out
    wantColumns1.remove('const')
    Sample(Variable1, DataFrame1, HowFill1, wantColumns1, VolumeOfSample1, out1)
    


def Sample(Variable, DataFrame, HowFill, WantColumns, VolumeOfSample, out):
    DistributionSample = SetOfTools[len(SetOfTools)-3].children[1].value
    df_S = DataFrame.copy()
    dataForAnalysis = df_S.dropna(subset=[Variable, ])
    yForAnalysis = dataForAnalysis.loc[:, Variable]
    if HowFill=='Среднее арифметическое':
        dataForAnalysis = dataForAnalysis.fillna(dataForAnalysis.mean())
    if HowFill=='Медиана':
        dataForAnalysis = dataForAnalysis.fillna(dataForAnalysis.median())  
    if HowFill=='Максимум':
         dataForAnalysis = dataForAnalysis.fillna(dataForAnalysis.max())
    if HowFill=='Минимум':
         dataForAnalysis = dataForAnalysis.fillna(dataForAnalysis.min())       
    xForAnalysis = dataForAnalysis.loc[:, WantColumns]
    dataForAnalysis = dataForAnalysis[WantColumns]    
    regressor = sm.OLS(list(yForAnalysis), add_constant(xForAnalysis)).fit()
    
    #     .value
    Columns = []
    leftBounds = {}
    rightBounds = {}
    Averages = {}
    Variances = {}
    
    
    for i in range(0, int((len(SetOfTools)-4)/3)):
        Columns.append(SetOfTools[1+3*i].value)
        leftBounds[SetOfTools[1+3*i].value] = SetOfTools[2+3*i].value
        rightBounds[SetOfTools[1+3*i].value] = SetOfTools[3+3*i].value
        Averages[SetOfTools[1+3*i].value] = (leftBounds[SetOfTools[1+3*i].value]+rightBounds[SetOfTools[1+3*i].value])/2
        Variances[SetOfTools[1+3*i].value] = rightBounds[SetOfTools[1+3*i].value]-(rightBounds[SetOfTools[1+3*i].value]+leftBounds[SetOfTools[1+3*i].value])/2
    
#     if Sample
    TestDataFrame = pd.DataFrame(columns=Columns)
    rng = np.random.RandomState(0)
    if DistributionSample != 'Нормальное':
        for i in range(0, int((len(SetOfTools)-4)/3)):
            TestDataFrame[SetOfTools[1+3*i].value] = rng.uniform(leftBounds[SetOfTools[1+3*i].value],
                                                                 rightBounds[SetOfTools[1+3*i].value], 
                                                                 VolumeOfSample)
    else:
        for i in range(0, int((len(SetOfTools)-4)/3)):
            TestDataFrame[SetOfTools[1+3*i].value] = rng.normal(Averages[SetOfTools[1+3*i].value], 
                                                                Variances[SetOfTools[1+3*i].value], 
                                                                VolumeOfSample)
            
        
    TestDataFrame['predict']=TestDataFrame['const']
    Columns.remove('const')
    for i in Columns:
        TestDataFrame['predict']=TestDataFrame['predict']+regressor.params[i]*TestDataFrame[i]
        
  
    data = TestDataFrame['predict']


    twentyfifth, seventyfifth, ninetyfifth = np.percentile(data, [10, 50, 90])
    TestDataFrame['predict'][TestDataFrame['predict']<0]=float('nan')
    TestDataFrame.dropna(subset=['predict', ])
    data = TestDataFrame['predict']
    


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
    counts, bins, patches = ax.hist(data, 
                                    facecolor=perc_50_colour, 
                                    edgecolor='gray', 
                                    bins=1+math.trunc(math.log2(VolumeOfSample)))


    # Set the ticks to be at the edges of the bins.
    ax.set_xticks(bins.round(2))
    plt.xticks(rotation=70)

    # Set the graph title and axes titles
    #plt.title('Distribution of randomly generated numbers', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.xlabel(Variable, fontsize=20)

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
    ax.annotate('N='+str(VolumeOfSample)
                +'\n 10 Percentile='+str(round(twentyfifth,4))
                +'\n 50 Percentile='+str(round(seventyfifth,4))
                +'\n 90 Percentile='+str(round(ninetyfifth,4)),
                xy=(.75,.50), xycoords='figure fraction',
                horizontalalignment='center', verticalalignment='bottom',
                fontsize=20, bbox=dict(boxstyle="round", fc="white"),
                rotation=0)
    plt.legend(handles, labels, bbox_to_anchor=(0.5, 0., 0.80, 0.99), fontsize=18)

    with out:
        print('Результаты сэмплирования')
        plt.show()
  

        
def main(out, df1):
    global df
    df=df1.copy()
    dfFloat = df.copy().loc[:, df.dtypes == float]
    dfStrings = df.copy().loc[:, df.dtypes != float]
#     Variables_VisualTab.options = list(df.columns)
    DependVariable_MainAlgo_ModelingTab.options = list(dfFloat.columns)
    ExcludeColumns_MainAlgo_ModelingTab.options = list(dfFloat.columns)
    Variable_Cluster_ModelingTab.options = list(dfFloat.columns)
    IndependVariable_Diskr_ModelingTab.options = list(dfStrings.columns)
    DependVariable_Diskr_ModelingTab.options = list(dfFloat.columns)       
    ButtonCorrelation_ModelingTab.on_click(functools.partial(Correlation_Click, out=out))
    Button_Diskr_ModelingTab.on_click(functools.partial(Diskr_Modeling_Click, out=out))
    Button_Cluster_ModelingTab.on_click(functools.partial(Cluster_Modeling_Click, out=out))
    ButtonStart_MainAlgo_ModelingTab.on_click(functools.partial(MainAlgo_Modeling_Click, out=out))
    return(Accordion_ModelingTab)
                                              
                                              

