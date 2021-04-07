import PySimpleGUI as sg
import cv2 as cv
import numpy as np
import imutils

"""
Demo program that displays a webcam using OpenCV
"""
def main():

    sg.theme('SystemDefault')
    Exercises = ['Shoulder Press', 'Squat', 'Bicep Curl']

    # define the window layout
    layout = [[sg.Text('Pose Estimation Workout', size=(40, 1), justification='center', font='Helvetica 20')],
              [sg.Listbox(Exercises, size=(20,4), pad=((250, 0), 3),key='_LIST_', enable_events=True)],
              [sg.Image(filename='', key='image'), sg.Image(filename='', key='image2')],
              [sg.Button('START', size=(10, 1),pad=((270, 0), 3), font='Helvetica 14'), sg.Exit('Exit', size=(10, 1),pad=((270, 0), 9), font='Helvetica 14')]]

    # create the window and show it without the plot
    window = sg.Window('Demo Application - OpenCV Integration', layout, location=(0,0))

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    '''cap = cv.VideoCapture(0)
    cap1 = cv.VideoCapture('abdultest.mp4')'''

    # recording = False

    while True:
        event, values = window.read()

        if sg.WIN_CLOSED or event == 'Exit':
        	window.close()
        	break

        if event == 'START':
            selection = values['_LIST_']
            what = type(selection)
            print(selection)
            print(what)

            if not selection:
                print('YES')


            # recording = True
        '''    window.Hide()
            layout2 = [[sg.Test('Window 2')],[sg.Exit('Exit')]]
            win2 = sg.Window('2nd window', layout2, location=(0,0))'''

        '''  if event == sg.WIN_CLOSED or event == 'Exit':
            	window.close()
            	window.UnHide()
            	break'''




        

        '''if recording:
            ret, frame = cap.read()
            ret, frame1 = cap1.read()
            
            frame1 = imutils.resize(frame1, width=640)
            #frame1 = cv.resize(frame1,(640, 400));

            imgbytes = cv.imencode('.png', frame)[1].tobytes()  # ditto
            window['image'].update(data=imgbytes)

            imgbytes = cv.imencode('.png', frame1)[1].tobytes()  # ditto
            window['image2'].update(data=imgbytes)'''
            
main()