# Lepton FLIR 3.5 based IR Tool Wear Monitor 

## Quick Description 
As part of my doctoral dissertation [Citation pending] and subsequent research publication [Citation pending]. I built this app to monitor for tool wear of an endmill in real time during milling operations. The goal was to monitor the temperature of the endmill while cutting a sample part [Dissertation citation] and then develop a linear regression model to establish a relationship between tool wear and temperature. The intended end goal is to develop a low-cost tool wear sensor that can be integrated in any system, but that is in the future. I took heavy inspiration from Ramirez-Nunez *et al.* [[3]](https://link.springer.com/article/10.1007/s00170-018-2060-4) but of course, took my own approach. 

## Camera Showcase 
The Lepton FLIR camera was chosen for this project since it is a low-cost, opensource IR camera with multiple modules ranging from arduino modules to USB modules. FLIR offers [dev kits](https://oem.flir.com/developer/lepton-family/) for LINUX, Windows and Raspberry Pi and code for the camera can be created using Python and MATLAB. FLIR even offers a [mobile SDK](https://oem.flir.com/developer/lepton-family/developer-mobile-sdk/) to develop IOs and Android mobile apps. [Third party modules](https://groupgets.com/collections/lepton) can also be used for other systems such as Arduino. In the case for the project the [PureThermal Mini Pro JST-SR](https://groupgets.com/products/purethermal-mini-pro-jst-sr) to connect the camera directly to a computer. The camera itself has a good enough resolution and temperature range for the application in question since the plas was to mount the camera directly on the spindle, please check the [camera description page](https://oem.flir.com/products/lepton/?model=500-0758-03&vertical=microcam&segment=oem) for more info.  
![lepton_pdp](https://github.com/user-attachments/assets/b41c1952-87d2-46bd-bae8-990dadeb8ad9)

## Code Description
The code produces an application using the [Tkinter](https://docs.python.org/3/library/tkinter.html) library which streams the IR image gathere from the Lepton FLIR 3.5 camera. A mask can be traced on top of the stream widget, this is intended to tell the edge detection algorithm where to detect the edges for the region of interest, the edge detection is done via the [OpenCV](https://opencv.org/) library. The average and maximum temperature are recorded over time once the "Start Recoding" button is press, the area of the ROI is also recorded since in literature it is mentioned that the temperature distribution along the endmill is an indicator of tool wear [[4]](https://www.mdpi.com/1424-8220/21/19/6687). A vertical line can also be traced, the purpose of the vertical line is to create 5 horizontal lines that act as ROIs and have the purpose of indicating the temperature distribution along the endmill. When the "Stop Recording" button is pressed then the app lets you save all the data into a CSV file. The lines are "upside down" because the camera was mounted that way in my project so the base of the endmill would be at the "top" of the image. 

## Installation Instructions
To intall the program is simple, just **clone** this git and follow the instructions from the **Jupyter** file inside the **Examples** directory but there are some recommendations that I have to make. First, the code is based on Python 3.13, so I would recommend sticking to this version. I have had problems with the **dlls** so I would strognly recommend creating a virtual environment for the application. Then is just a matter of installing the necessary libraries not described in the jupyter file such as Tkinter and OpenCV. If a virtual enviroment was created please make sure to install the libraries inside the environment. Usually Tkinter is already included in Python 3 by default but you can check in your terminal using:
```
python -m tkinter
```
To install OpenCV just run in your terminal:
```
pip install opencv-python
```
## Contributions
If you want to contribute to the code you can create your own branch just send me a message at juanp.gchavira@gmail.com to talk and know what are you planning to do with this code. Ofcourse if you're planning on using this code in your research, please cite my work. Don't be afraid to reach out and give me any feedback or suggestion I can also answer questions. It would be cool to use this code to gather enough data to build a machine learning model to predict tool wear using temperature, If you're also interested in that pleae reach out. 

## References 
[1]  
[2]  
[3] J. A. Ramirez-Nunez, M. Trejo-Hernandez, R. J. Romero-Troncoso, G. Herrera-Ruiz, and R. A. Osornio-Rios, “Smart-sensor for tool-breakage detection in milling process under dry and wet conditions based on infrared thermography,” Int. J. Adv. Manuf. Technol., vol. 97, no. 5, Art. no. 5, July 2018, [doi: 10.1007/s00170-018-2060-4](https://link.springer.com/article/10.1007/s00170-018-2060-4)  
[4] N. Brili, M. Ficko, and S. Klančnik, “Tool Condition Monitoring of the Cutting Capability of a Turning Tool Based on Thermography,” Sensors, vol. 21, no. 19, Art. no. 19, Jan. 2021, [doi: 10.3390/s21196687](https://www.mdpi.com/1424-8220/21/19/6687).






