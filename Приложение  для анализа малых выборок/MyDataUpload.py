#Перезагрузка ядра Jupyter Notebook для корректной визуализации интерфейса.
from IPython.core.display import HTML
HTML("<script>Jupyter.notebook.kernel.restart()</script>")

#Импорт стандартных библиотек. 
import functools #используется функция partial

#Импорт сторонних библиотек. 
#Импорт библиотки для операций с датафреймом.
import pandas as pd
#Импорт библиотеки для интерфейса.
import ipywidgets as widgets



df = pd.DataFrame()

#----------------------------------------------------------------------------------------
#Data upload tab--------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#Button of file uploader
FileUploader_UploadTab = widgets.FileUpload(
    accept='.xlsx, .csv',  
    multiple=True,  # True to accept multiple files upload else False
    description='Файл',
    button_style='success'
) 


#Text of row range, which you want use in file
TextRows_UploadTab = widgets.Text(
    value='2-400',
    placeholder='Type something',
    description='Строки:'
)


#Text of column range, which you want use in file
TextColumns_UploadTab = widgets.Text(
    value='0-400',
    placeholder='Type something',
    description='Столбцы:'
)


#Int of main line number (with names of columns)
IntHeader_UploadTab = widgets.IntText(
    value='0',
    description='Номер строки с названиями переменных:',
    style={'description_width': 'initial'}
)


#Int of main line number (with names of columns)
IntHowRowsDisplay_UploadTab = widgets.IntText(
    value='5',
    description='Количество строк для вывода:',
    style={'description_width': 'initial'}
)


#Int of main line number (with names of columns)
TextWhatIsNan_UploadTab = widgets.Text(
    value='-',
    placeholder='Type something',
    description='Символ пустой ячейки:',
    style={'description_width': 'initial'}
)


#Button of data frame upload
ButtonUpload_UploadTab = widgets.Button(
    description='Загрузить данные'
)


#function for upload data frame with some numbers of rows
def DataFrameCreate(UseRows_DFC):
    #reading data from the interface
    [UploadedFile_DFC] = FileUploader_UploadTab.value
    dfAll_DFC = pd.read_excel(UploadedFile_DFC) #for help
    
    #searching NaN
    WheredashColumns_DFC = TextColumns_UploadTab.value.find('-')

    UseColumns_DFC = [i for i in range(int(TextColumns_UploadTab.value[0:WheredashColumns_DFC]), 
                                   int(TextColumns_UploadTab.value[WheredashColumns_DFC+1:])+1)]
    AllRows_DFC = [i for i in range(1, len(dfAll_DFC.index)+1)]
    NoUseRows_DFC = list(set(AllRows_DFC) - set(UseRows_DFC))
    df_DFC = pd.read_excel(UploadedFile_DFC, usecols=UseColumns_DFC, skiprows=NoUseRows_DFC, 
                       header=IntHeader_UploadTab.value)   

    return(df_DFC)


#click function for upload data frame
def DataFrameCreate_Click(b, out=widgets.Output()):
    global df
    #upload file with corresponding number of rows and columns
    WheredashRows_DFCC = TextRows_UploadTab.value.find('-')
    UseRows_DFCC =[i for i in range(int(TextRows_UploadTab.value[0:WheredashRows_DFCC]), 
                               int(TextRows_UploadTab.value[WheredashRows_DFCC+1:])+1)] 
    df_DFCC = DataFrameCreate(UseRows_DFCC)
  
    #upload data frame with first row to define types of data in columns
    df0_DFCC = DataFrameCreate([1])  
    Typesdf0_DFCC = dict(df0_DFCC.dtypes)
            
    NanSymbol_DFCC = TextWhatIsNan_UploadTab.value
    WhoObjectIn_df0_DFCC = list(df0_DFCC.select_dtypes(include=['object']).columns)
    WhoFloatIn_df0_DFCC = list(df0_DFCC.select_dtypes(include=['float64']).columns)
            
    for i in WhoObjectIn_df0_DFCC:
        df_DFCC[i] = df_DFCC[i].replace(NanSymbol_DFCC,  str('nan'))
        #wtf, but it works
        df_DFCC[i] = df_DFCC[i].astype(str) 
        df_DFCC[i] = pd.Series(df_DFCC[i], dtype='string')   

    for i in WhoFloatIn_df0_DFCC:
        df_DFCC[i] = df_DFCC[i].replace(NanSymbol_DFCC,  float('nan'))
        #wtf, but it works
        df_DFCC[i] = df_DFCC[i].astype(float) 
        df_DFCC[i] = pd.Series(df_DFCC[i], dtype='float64')            
       
    with out:
        print('Данные загружены')
        display(df_DFCC.head(IntHowRowsDisplay_UploadTab.value))
        df_DFCC.info()
    df = df_DFCC.copy()
     
    
#Box of data frame upload
Box_UploadTab = widgets.VBox(
    children=(FileUploader_UploadTab,
              TextRows_UploadTab,
              TextColumns_UploadTab,
              IntHeader_UploadTab,
              TextWhatIsNan_UploadTab,
              IntHowRowsDisplay_UploadTab,
              ButtonUpload_UploadTab
             )
)


def main(out):    
    ButtonUpload_UploadTab.on_click(functools.partial(DataFrameCreate_Click, out=out))
    return([Box_UploadTab, df])

def IfButtonUploadClicked():
    global df
    key_IBUC = False
    if df.empty == False:
        key_IBUC = True
    
    return(key_IBUC)
