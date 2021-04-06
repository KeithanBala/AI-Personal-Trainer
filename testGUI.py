import PySimpleGUI as sg
import os

def shoulder_press():
    
    layout = [[sg.Text('This is the shoulder press workout')], [sg.Button('Start'),sg.Exit('Exit')]]
    window = sg.Window('Shoulder Press Workout', layout, modal = True)

    while True:
        event, values = window.read()
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break

        if event == 'Start':
            os.system('python run_webcamSplitScreen_angle.py --model=mobilenet_thin --resize=432x368 --camera=0')


    
    window.close()

def squat():
    
    layout = [[sg.Text('This is the squat workout')], [sg.Button('Start'),sg.Exit('Exit')]]
    window = sg.Window('Squat Workout', layout, modal = True)

    while True:
        event, values = window.read()
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break
    
    window.close()

def lateral_raise():
    
    layout = [[sg.Text('This is the Lateral Raise workout')], [sg.Button('Start'),sg.Exit('Exit')]]
    window = sg.Window('Lateral Raise Workout', layout, modal = True)

    while True:
        event, values = window.read()
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break
    
    window.close()

def main_menu():

    layout = [[sg.Text('Pose Estimation Workout', size=(40, 1), justification='center', font='Helvetica 20')],
              [sg.Button('Shoulder Press'), sg.Button('Lateral Raise'), sg.Button('Squat'), sg.Exit('Exit', size=(10, 1),pad=((270, 0), 9), font='Helvetica 14')]]
    
    window = sg.Window('MAIN MENU', layout)

    while True:
        event, values = window.read()

        if event == 'Exit' or event == sg.WIN_CLOSED:
            window.close()
            break

        if event == 'Shoulder Press':
            shoulder_press()
        
        if event == 'Lateral Raise':
            lateral_raise()

        if event == 'Squat':
            squat()
        

if __name__ == "__main__":
    main_menu()
