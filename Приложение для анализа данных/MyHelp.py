#Перезагрузка ядра Jupyter Notebook для корректной визуализации интерфейса.
from IPython.core.display import HTML
HTML("<script>Jupyter.notebook.kernel.restart()</script>")

#Библиотека для интерфейса (сторонняя).  
import ipywidgets as widgets
from ipywidgets import Layout



#----------------------------------------------------------------------------------------
#Help data tab--------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
HelpArea = widgets.Output(layout={'border': '1px solid black'})

video_instruction = widgets.Button(
    description='Видеоинструкция',
    button_style='info'
)

def video_instruction_click(b):
    HelpArea.clear_output()
    
    video = widgets.Video.from_file('VideoInstruction.mp4')
    with HelpArea:
        display(video) 
  
    
video_instruction.on_click(video_instruction_click)

instruction = widgets.Button(
    description='Инструкция',
    button_style='info'
)

def instruction_click(b):
    HelpArea.clear_output()
    sample = open('Instruction.txt') 
    with HelpArea:
        print(sample.read()) 
    sample.close()   

instruction.on_click(instruction_click)

about = widgets.Button(
    description='О программе',
    button_style='info'
)

def about_click(b):
    HelpArea.clear_output()
    sample = open('About.txt') 
    with HelpArea:
        print(sample.read()) 
    sample.close()  
    
about.on_click(about_click)


#Box in the about tab
Box_Help = widgets.VBox(
    children=(video_instruction, instruction, about, HelpArea) 
)


def main():
    return(Box_Help)







