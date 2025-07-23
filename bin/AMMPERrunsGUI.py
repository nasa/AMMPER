# This Python file uses utf-8 encoding.

"""
Graphical User Interface for AMMPER Runs v2.0

Created by Madeline Marous, in coordination with original code created by Amrita Singh and edited by Daniel Palacios.
Review README and Credits for more information.

"""
# GUI modules
import sys
from time import sleep
import subprocess
import random
# from AMMPERBulk_aB import finish

from PyQt5.QtWidgets import QApplication, QWidget

# AMMPER modules
import numpy as np
import random as rand
import uuid as uuid
from cellDefinition import Cell
from genTraverse_groundTesting import genTraverse_groundTesting
from genTraverse_deepSpace import genTraverse_deepSpace
from genROS import genROS
from genROSOld import genROSOld
from cellPlot import cellPlot
from cellPlot_deepSpace import cellPlot_deepSpace
from GammaRadGen import GammaRadGen
import os
import time
import pandas as pd
import time
from sklearn.model_selection import train_test_split
start_time = time.time()
from formgui import Ui_Widget

class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)
# Confirming connection from UI layout.
        self.stackedWidget = self.ui.stackedWidget
        self.stackedWidget.setCurrentIndex(0)

        self.display = True

        self.radioButton = self.ui.radioButton
        self.radioButton_2 = self.ui.radioButton_2
        self.radioButton_3 = self.ui.radioButton_3
        self.radioButton_4 = self.ui.radioButton_4
        self.radioButton_5 = self.ui.radioButton_5
        self.radioButton_6 = self.ui.radioButton_6
        self.radioButton_7 = self.ui.radioButton_7
        self.radioButton_8 = self.ui.radioButton_8

        self.pushButton = self.ui.pushButton
        self.pushButton_2 = self.ui.pushButton_2
        self.pushButton_3 = self.ui.pushButton_3
        self.pushButton_4 = self.ui.pushButton_4
        self.pushButton_5 = self.ui.pushButton_5
        self.pushButton_6 = self.ui.pushButton_6
        self.pushButton_7 = self.ui.pushButton_7
        self.pushButton_8 = self.ui.pushButton_8
        self.pushButton_9 = self.ui.pushButton_9
        self.pushButton_10 = self.ui.pushButton_10
        self.pushButton_11 = self.ui.pushButton_11

        self.dial = self.ui.dial
        self.dial.valueChanged.connect(self.dialChange)

        self.verticalSlider = self.ui.verticalSlider
        
        self.progressBar = self.ui.progressBar

        self.plainTextEdit = self.ui.plainTextEdit

        self.label = self.ui.label
        self.label_2 = self.ui.label_2
        self.label_3 = self.ui.label_3
        self.label_4 = self.ui.label_4
        self.label_5 = self.ui.label_5
        self.label_6 = self.ui.label_6
        self.label_7 = self.ui.label_7
        self.label_8 = self.ui.label_8
        self.label_9 = self.ui.label_9
        self.label_10 = self.ui.label_10
        self.label_11 = self.ui.label_11
        self.label_12 = self.ui.label_12
        self.label_13 = self.ui.label_13
        self.label_14 = self.ui.label_14
        self.label_15 = self.ui.label_15
        self.label_16 = self.ui.label_16
        self.label_17 = self.ui.label_17
        self.label_18 = self.ui.label_18
        self.label_19 = self.ui.label_19
        self.label_20 = self.ui.label_20
        self.label_21 = self.ui.label_21
        self.label_22 = self.ui.label_22
        self.label_23 = self.ui.label_23
        self.label_24 = self.ui.label_24
        self.label_25 = self.ui.label_25
        self.label_26 = self.ui.label_26
        self.label_27 = self.ui.label_27
        self.label_28 = self.ui.label_28
        self.label_29 = self.ui.label_29
        self.label_30 = self.ui.label_30
        self.label_31 = self.ui.label_31
        self.label_32 = self.ui.label_32
        self.label_33 = self.ui.label_33
        self.label_34 = self.ui.label_34
        self.label_35 = self.ui.label_35
        self.label_36 = self.ui.label_36
        self.label_37 = self.ui.label_37
        self.label_38 = self.ui.label_38
        self.label_39 = self.ui.label_39
        self.label_40 = self.ui.label_40
        self.label_41 = self.ui.label_41
        self.label_42 = self.ui.label_42
        self.label_43 = self.ui.label_43
        self.label_44 = self.ui.label_44
        self.label_45 = self.ui.label_45
        self.label_46 = self.ui.label_46
        self.label_47 = self.ui.label_47

        self.checkBox = self.ui.checkBox
        self.checkBox_2 = self.ui.checkBox_2
        self.checkBox_3 = self.ui.checkBox_3
        self.checkBox_4 = self.ui.checkBox_4
        self.checkBox_5 = self.ui.checkBox_5
        self.checkBox_6 = self.ui.checkBox_6
        self.checkBox_7 = self.ui.checkBox_7
        self.checkBox_8 = self.ui.checkBox_8

        self.pushButton_8 = self.ui.pushButton_8
        self.pushButton_8.setEnabled(False)

        self.plainTextEdit = self.ui.plainTextEdit
        self.path = self.plainTextEdit.toPlainText()
        self.plainTextEdit.setPlainText("/Results") #hopefully I got this right
        self.plainTextEdit.setStyleSheet("color: black;")

        self.ui.groupBox.setStyleSheet("color: white;")
        self.ui.groupBox_2.setStyleSheet("color: white;")
        self.ui.groupBox_3.setStyleSheet("color: white;")        
        self.ui.groupBox_4.setStyleSheet("color: white;")
        self.ui.groupBox_5.setStyleSheet("color: white;")
        self.ui.groupBox_6.setStyleSheet("color: white;")
        self.ui.groupBox_7.setStyleSheet("color: white;")

        # Connecting GUI framework to AMMPER code.
        self.ui.pushButton.clicked.connect(self.pushButton_clicked) # Launch GUI
        self.ui.pushButton_2.clicked.connect(self.pushButton_2_clicked) # Launch CLI
        self.ui.pushButton_3.clicked.connect(self.pushButton_3_clicked) # Credits (TBD)
        self.ui.pushButton_4.clicked.connect(self.pushButton_4_clicked) # Set Up Simulation
        self.ui.pushButton_5.clicked.connect(self.simSetup) # Run Simulation
       # self.ui.pushButton_5.clicked.connect(self.dataSetup) # Add another page
        self.ui.pushButton_5.clicked.connect(self.dataRun) 
        self.ui.pushButton_7.clicked.connect(self.fileExport) #fix this?
        self.ui.pushButton_8.clicked.connect(self.resultsScreen)

        self.ui.pushButton_5.clicked.connect(self.toggle_visibility)
        self.ui.pushButton_9.clicked.connect(self.toggle_visibility1)
        self.ui.pushButton_10.clicked.connect(self.toggle_visibility2)
        self.ui.pushButton_11.clicked.connect(self.toggle_visibility3)
        self.ui.pushButton_15.clicked.connect(self.goBack)

        # Connecting radio buttons.
        self.radioButton.toggled.connect(self.onRadioButtonClicked)
        self.radioButton_2.toggled.connect(self.onRadioButtonClicked)
        self.radioButton_3.toggled.connect(self.onRadioButtonClicked)
        self.radioButton_4.toggled.connect(self.onRadioButtonClicked)
        
        self.radioButton_5.toggled.connect(self.onRadioButtonClicked2)
        self.radioButton_6.toggled.connect(self.onRadioButtonClicked2)
        
        self.radioButton_7.toggled.connect(self.onRadioButtonClicked3)
        self.radioButton_8.toggled.connect(self.onRadioButtonClicked3)

        self.checkBox.stateChanged.connect(self.fileExport)
        self.checkBox_2.stateChanged.connect(self.fileExport)

        self.verticalSlider.valueChanged.connect(self.Slider)
        self.slider = self.verticalSlider.value()

        self.label_41.setVisible(False)
        self.label_45.setVisible(False)
        self.label_46.setVisible(False)
        self.label_47.setVisible(False)

        self.setWindowTitle("AMMPERruns")

        # Initializing variables.

        self.radAmount = 0.0
        self.cellType = "" 
        self.radType = ""  
        self.N = 0 
        self.gen = 0
        self.ROSType = ""
        self.simAmt = 1
        self.Gy = float(0)
        self.simDescription = ""
        self.downloadHelp = ''
        self.down2 =''
        self.down3 = ''
        self.sliderOn = True
        self.typeCount = 1
        self.instruct = ""
        self.simNum = 1
        self.plotCount = 0
        self.simCount = 0
        # Starting booleans
        self.display = False
        self.fileWritten = False
        self.doPlot = False
        self.doSim = False
        self.doData = False
        self.radData = np.zeros([1,6],dtype = float)
        self.ROSData = np.zeros([1,6],dtype = float)


    def pushButton_clicked(self):
        self.stackedWidget.setCurrentIndex(1)

    def pushButton_2_clicked(self):
        i = 1
        while i == 1:     
            QApplication.exit()
            choiceA = input("Hello, and welcome to the AMMPER Runs CLI. Features may be limited. Press 'a' for regular testing, 'b' for gamma, or 'c' to quit.")
            if choiceA == 'a':
                print("Regular testing chosen.")
                subprocess.call(["python", "AMMPERruns_aB.py"]) # Launch CLI
                print("Please enter the following prompts to finish setting up your simulation.")
                self.simNum = int(input("Enter the amount of runs you would like to complete today."))
                print("Runs: ", self.simNum)
                cellSelect = input("Please enter cell type:\n\ta)Wild Type\n\tb)rad51\n")
                if ROSSelect == 'a':
                    labelROS = 'WT'
                elif ROSSelect == 'b':
                    labelROS = 'rad51'
                ROSSelect = input("Please enter ROS Model: \n\ta)Basic ROS\n\tb)Complex ROS\n")
                radSelect = input("Please enter radiation dose. Options are: 0, 2.5, 5, 10, 20, 30 (Gy).\n")
                if ROSSelect == 'a':
                    labelROS = 'Basic'
                elif ROSSelect == 'b':
                    labelROS = 'Complex'
                self.instruct = "python .\AMMPERBulk_aBmodule.py d " + cellSelect + ' ' + ROSSelect + ' ' + radSelect + ' ' + labelCell + '_' + labelROS + '_' + radSelect
            elif choiceA == 'b': 
                print("Gamma testing chosen.")
                print("Please enter the following prompts to finish setting up your simulation.")
                self.simNum = int(input("Enter the amount of runs you would like to complete today."))
                print("Runs: ", self.simNum)
                cellSelect = input("Please enter cell type:\n\ta)Wild Type\n\tb)rad51\n")
                if ROSSelect == 'a':
                    labelCell = 'WT'
                elif ROSSelect == 'b':
                    labelCell = 'rad51'
                ROSSelect = input("Please enter ROS Model: \n\ta)Basic ROS\n\tb)Complex ROS\n")
                radSelect = input("Please enter radiation dose. Options are: 0, 2.5, 5, 10, 20, 30 (Gy).\n")
                if ROSSelect == 'a':
                    labelROS = 'Basic'
                elif ROSSelect == 'b':
                    labelROS = 'Complex'
                self.instruct = "python .\AMMPERBulkGAMMAFINAL.py d " + cellSelect + ' ' + ROSSelect + ' ' + radSelect + ' ' + labelCell + '_' + radSelect
                subprocess.call(["python", "AMMPERrunsGAMMAFINALmodule_aB.py"]) # Launch CLI
            elif choiceA == 'c':
                i = 0
            else:
                print("Invalid input. Please try again.")

    def pushButton_3_clicked(self):
        self.stackedWidget.setCurrentIndex(4)

    def pushButton_4_clicked(self):
        if self.display:
            self.stackedWidget.setCurrentIndex(2)
            self.progressBar.setValue(0)
            self.simSetup()
        else: 
            self.stackedWidget.setCurrentIndex(2)
            QApplication.exit()

    """def pushButton_5_clicked(self):
        new_page = QWidget(self.stacked_widget_2)
        page_name = f"Page {self.page_counter}"
        new_page.setObjectName(page_name)
        self.stacked_widget.addWidget(new_page)
        self.page_counter += 1
        new_page_index = self.stacked_widget.count() - 1
        self.stacked_widget.setCurrentIndex(new_page_index)
        self.typeCount = self.typeCount + 1"""
    
    def goBack(self):
        self.stackedWidget.setCurrentIndex(0)

    def onRadioButtonClicked(self):
        if self.radioButton.isChecked():
            self.downloadHelp = 'a'
            self.sliderOn = True
            self.Gy = float(self.radAmount)
            self.radType = "150 MeV Proton"
            self.gen = 15
            self.radGen = 2
            self.N = 64
            if self.Gy == 0:
                self.radData = np.zeros([1,6],dtype = float)
                self.ROSData = np.zeros([1,6],dtype = float)
            elif self.radioButton_2.isChecked():
                self.downloadHelp = 'b'
                self.sliderOn = False
                self.radType = "GCRSim"
                self.gen = 15
                self.radGen = 2
                self.N = 64
                self.radData = np.zeros([1,6],dtype = float)
                self.ROSData = np.zeros([1,6],dtype = float)
            elif self.radioButton_4.isChecked():
                self.downloadHelp = 'c'
                self.sliderOn = False
                self.radType = "Deep Space"
                self.gen = 15
                self.N = 300
                self.radGen = 0
                self.Gy = 0
                self.radData = np.zeros([1,6],dtype = float)
                self.ROSData = np.zeros([1,6],dtype = float)
            elif self.radioButton_3.isChecked():
                self.downloadHelp = 'd'
                self.sliderOn = False
                self.radType = "Gamma"
                self.gen = 15
                self.radGen = 10
                self.N = 64
                self.radData = np.zeros([1,6],dtype = float)
                self.ROSData = np.zeros([1,6],dtype = float)

    def Slider(self):
        if self.sliderOn == True:
            self.slider = self.verticalSlider.value()
            if self.slider == 0:
                self.radAmount = 0
            elif self.slider == 1:
                self.radAmount = 2.5
            elif self.slider == 2:
                self.radAmount = 5
            elif self.slider == 3:
                self.radAmount = 10
            elif self.slider == 4:
                self.radAmount = 20
            elif self.slider == 5:
                self.radAmount = 30
        elif self.sliderOn == False:
            self.horizontalSlider.setEnabled(False)
        self.Gy = float(self.radAmount)
        self.label_43.setText(str(self.radAmount))

    def toggle_visibility(self):
        current_visibility = self.label_41.isVisible()
        self.label.setVisible(not current_visibility)

    def toggle_visibility1(self):
        current_visibility = self.label_45.isVisible()
        self.label.setVisible(not current_visibility)

    def toggle_visibility2(self):
        current_visibility = self.label_46.isVisible()
        self.label.setVisible(not current_visibility)

    def toggle_visibility3(self):
        current_visibility = self.label_47.isVisible()
        self.label.setVisible(not current_visibility)

    def dialChange(self):
        self.simAmt = self.dial.value()
        self.label_21.setText(str(self.simAmt))

    def onRadioButtonClicked2(self):
        if self.radioButton_5.isChecked():
            self.cellType = "wt"
            self.down2 = 'a'
        elif self.radioButton_6.isChecked():
            self.cellType = "rad51"
            self.down2 = 'b'

    def onRadioButtonClicked3(self):
        if self.radioButton_7.isChecked():
            self.ROSType = "Basic ROS"
            self.down3 = "Basic"
        elif self.radioButton_8.isChecked():
            self.ROSType = "Complex ROS"
            self.down3 = "Complex"

    def fileExport(self):
        if self.checkBox.isChecked():
            self.path = self.plainTextEdit.toPlainText()
            self.fileWritten = True
            try:
                with open(self.path, 'r'):
                    self.label_5.setText("Valid path!")
            except:
                self.label_5.setText("Invalid path.")
                self.label_5.setStyleSheet("color: red;")
        if self.checkBox_2.isChecked():
            self.display = True

    def dataSetup(self):
        if self.checkBox_3.isChecked():
            self.doData = True
            self.setEnabled.checkBox_4(True)
        else:
            self.doData = False
        if self.doData == False:
            self.setEnabled.checkBox_4(False)
        if self.checkBox_4.isChecked():
            self.doPlot = True
            self.plotCount = self.plotCount + 1
        else:
            self.doPlot = False
        if self.checkBox_5.isChecked():
            self.doSim = True
            self.simCount = self.simCount + 1
        else:
            self.doSim = False

    #self.start_time = self.time.time()

    def dataRun(self):
        if self.doData:
            try:
                if radType == "Gamma":
                    fileName1 = "AMMPERBulk_GAMMAfinal.py"
                    fileName2 = "AMMPERruns_GAMMAFINALmodule.py"
                    letter1 = 'd'
                    letter2 = self.down2
                    letter3 = self.down3
                    underA = ""
                    typeB = ""
                else:
                    fileName1 = "AMMPERBulk_aB.py"
                    fileName2 = "AMMPERruns_aBmodule.py"
                    letter1 = self.downloadHelp
                    letter2 = self.down2
                    letter3 = self.down3
                    underA = "_"
                    typeB = self.ROSModel
            except:
                self.stackedWidget.setCurrentIndex(0)
                self.label_2.setLabelText("Unavailable. Please try again.")
                self.label_2.setStyleSheet("color: red;")
            if self.doData:
                self.instruct = "python .\\" + fileName1 + " " + letter1 + " " + letter2 + ' ' + letter3 + ' ' + self.Gy + ' ' + self.radType + underA + typeA + '_' + self.Gy
                subprocess.call(["python", fileName2]) # Launch CLI
            self.label_20.setLabelText(finish)
            self.label_19.setText(self.radType)
            self.fact_list = ["AMMPER incorporates data from BioSentinel, the first biological CubeSat to fly beyond Low Earth Orbit (LEO).",
                            "AMMPER models after yeast (Saccharomyces cerevisiae) cells because they are eukaryotic, therefore similar in biology to human cells.", 
                            "AMMPER was first created by Amrita Singh in 2021.", "AMMPER is currently available for download under an open-source agreement.", 
                            "AMMPER is the first microbial model (to our knowledge) to study the effects of deep-space radiation on individual cells."]
            self.random_item = random.choice(self.fact_list)
            self.label_18.setStyleSheet("font-size: 11px;") 
            self.label_18.setText("Did you know: " + self.random_item)

  #  def plotRun(self)
   #     if self.doPlot


    def simSetup(self):
        if self.doSim:
            self.simDescription = "Cell Type: " + self.cellType + "\nRad Type: " + self.radType + "\nSim Dim: " +  str(self.N) + "microns\nNumGen: " + str(self.gen) + "ROS model: " + str(self.ROSType)

        # results folder name with the time that the simulation completed
            resultsName = time.strftime('%m-%d-%y_%H-%M') + "/"
            # determine path that all results will be written to
            resultsFolder = "Results/"
            #currPath = os.path.dirname("AMMPER")
            allResults_path = os.path.join(self.path,resultsFolder)
            currResult_path = os.path.join(allResults_path,resultsName)
            plots_path = os.path.join(currResult_path,"Plots/")

        # if any of the folders do not exist, create them
            if not os.path.isdir(resultsFolder):
                os.makedirs(resultsFolder)
            if not os.path.isdir(currResult_path):
                os.makedirs(currResult_path)
            if not os.path.isdir(plots_path):
                os.makedirs(plots_path)

            # write description to file
            np.savetxt(currResult_path+'simDescription.txt',[self.simDescription],fmt='%s')

        # SIMULATION SPACE INITIALIZATION

        # cubic space (0 = no cell, 1 = healthy cell, 2 = damaged cell, 3 = dead cell)
            T = np.zeros((self.N,self.N,self.N),dtype = int)
        # END OF SIMULATION SPACE INITIALIZATION
        # CELL INITIALIZATION

        # first cell is at center and is healthy
            firstCellPos = [int(self.N/2),int(self.N/2),int(self.N/2)]
            initCellHealth = 1

        # cells = list of cells - for new cells, cells.append
            # firstCellPos = initCellPos[0,:]
            firstUUID = uuid.uuid4()
            firstCell = Cell(firstUUID,firstCellPos,initCellHealth,0,0,0,0)
            T[firstCellPos[0],firstCellPos[1],firstCellPos[2]] = firstCell.health
            cells = [firstCell]
            # data: [generation, cellPosition, cellHealth]
            data = [0,firstCell.position[0],firstCell.position[1],firstCell.position[2],firstCell.health]    


        # END OF CELL INITIALIZATION

        #placeholder initialization
            if self.radType == "Deep Space":
                radData = np.zeros([1,7],dtype = float)
                ROSData = np.zeros([1,7],dtype = float)

            self.label_20.setText("Simulation beginning.")
            current_value = 0
            for g in range(1,self.gen + 1):
                self.label_2.setText("Generation " + str(g))
                current_value = self.progress_bar.value()
                self.progressBar.setValue(current_value + 5)

            self.fact_list = ["AMMPER incorporates data from BioSentinel, the first biological CubeSat to fly beyond Low Earth Orbit (LEO).",
                            "AMMPER models after yeast (Saccharomyces cerevisiae) cells because they are eukaryotic, therefore similar in biology to human cells.", 
                            "AMMPER was first created by Amrita Singh in 2021.", "AMMPER is currently available for download under an open-source agreement.", 
                            "AMMPER is the first microbial model (to our knowledge) to study the effects of deep-space radiation on individual cells."]
            self.random_item = random.choice(self.fact_list)
            self.label_19.setText(self.radType) # + ", " + self.typeCount + "test(s) total.") Eventually this will be added.
            self.label_18.setStyleSheet("font-size: 11px;") 
            self.label_18.setText("Did you know: " + self.random_item)

            if self.radType == "Gamma":
                if g == self.radGen:
                    
                    dose = 1
                    # radData = np.zeros([1, 6], dtype=float)
                    # Dose input, radGenE stop point for gamma radiation.
                    radData = GammaRadGen(dose)
                    # radData = np.delete(radData, (0), axis=0)
                    
                    if self.ROSType == "Complex ROS":
                        ROSData = genROS(radData, cells)
                    if self.ROSType == "Basic ROS":
                        ROSData = genROSOld(radData, cells)
            
                
            if self.radType == "150 MeV Proton":
                if g == self.radGen:
                    protonEnergy = 150
                    # these fluences are pre-calculated to deliver the dose to the volume of water
                    if self.Gy != 0:
                        if self.Gy == 2.5:
                            trackChoice = [1]
                            energyThreshold = 0
                        elif self.Gy == 5:
                            trackChoice = [1,1]
                            energyThreshold = 0
                        elif self.Gy == 10:
                            trackChoice = [1,1,1,1]
                            energyThreshold = 0
                        elif self.Gy == 20:
                            trackChoice = [1,1,1,1,1,1,1,1]
                            energyThreshold = 0
                        elif self.Gy == 30:
                            trackChoice = [1,1,1,1,1,1,1,1,1,1,1,1]
                            energyThreshold = 0
                        
                        # placeholder initialization - will hold information on all radiation energy depositions
                        radData = np.zeros([1,6],dtype = float)
                        # ROSData = np.zeros([1,6],dtype = float)

                        for track in trackChoice:
                            trackNum = track
                            # creates a traverse for every track in trackChoice
                            radData_trans = genTraverse_groundTesting(self.N,protonEnergy,trackNum,energyThreshold,self.radType)
                            # compile all energy depositions from individual tracks together
                            radData = np.vstack([radData,radData_trans])
                        
                        #remove placeholder of 0s from the beginning of radData
                        radData = np.delete(radData,(0),axis = 0)
                                    
                        # direct energy results in ROS generation - use energy depositions to calculate ROS species
                        if self.ROSType == "Complex ROS":
                            ROSData = genROS(radData,cells)
                        if self.ROSType == "Basic ROS":
                            ROSData = genROSOld(radData,cells)
                        #
                        # ROSData = np.delete(ROSData, (0), axis = 0)
                        
            elif self.radType == "Deep Space": 
                
                # take information from text file for #tracks of each proton energy
                deepSpaceFluenceDat = np.genfromtxt('DeepSpaceFluence0.1months_data.txt')
                # NOTE: For Deep Space sim, radiation delivery is staggered over time
                for it in range(len(deepSpaceFluenceDat)):
                    # get generation at which traversal will occur
                    currG = int(deepSpaceFluenceDat[it,5])
                    # determine how many traversals occur at this generation
                    numTrav = int(deepSpaceFluenceDat[it,4])
                    if g == currG:
                        # determine what proton energy the traversal has
                        protonEnergy = deepSpaceFluenceDat[it,0]
                        # parameter that allows non-damaging energy depositions to be ignored (used to speed up simulation)
                        energyThreshold = 20
                        for track in range(numTrav):
                            # choose a random track out of the 8 available/proton energy
                            trackNum  = int(rand.uniform(0,7))
                            # generate traversal data for omnidirectional traversals
                            radData_trans = genTraverse_deepSpace(self.N,protonEnergy,trackNum,energyThreshold)
                            # generate ROS data from the traversal energy deposition
                            # ROSData_new = genROS(radData_trans,cells)
                            if self.ROSType == "Complex ROS":
                                ROSData_new = genROS(radData_trans, cells)
                            if self.ROSType == "Basic ROS":
                                ROSData_new = genROSOld(radData_trans, cells)
                            
                            # creates a column indicating what generation the ROS and radData occured at
                            genArr = np.ones([len(radData_trans),1],dtype=int)*g
                            # compile radData with the generation indicator
                            radData_trans = np.hstack((radData_trans,genArr))
                            # compile radData from this traversal with all radData
                            radData = np.vstack([radData,radData_trans])
                            
                            # compile ROSData with the generation indicator
                            ROSData_new = np.hstack((ROSData_new,genArr))
                            #compile ROSData with all ROSData
                            ROSData = np.vstack([ROSData,ROSData_new])
                            
            elif self.radType == "GCRSim":
                if g == self.radGen:
                    # placeholder initialization
                    radData = np.zeros([1,6],dtype = float)
                    # take information from text file on traversals that will occur
                    GCRSimFluenceDat = np.genfromtxt('GCRSimFluence_data.txt',skip_header = 1)
                    for it in range(len(GCRSimFluenceDat)):
                        # for every traversal, get the proton energy of it
                        protonEnergy = int(GCRSimFluenceDat[it,0])
                        # parameter that allows non-damaging energy depositions to be ignored (used to speed up simulation)
                        energyThreshold = 20
                        # choose a random track out of the 8 available/proton energy
                        trackNum = int(rand.uniform(0,7))
                        # generate traversal data for unidirectional traversals
                        radData_trans = genTraverse_groundTesting(self.N,protonEnergy,trackNum,energyThreshold,self.radType)
                        # compile radData from this traversal with all radData
                        radData = np.vstack([radData,radData_trans])
                        
                        #remove placeholder from beginning
                        radData = np.delete(radData,(0),axis = 0)
                    # generate ROS data from all traversal energy depositions
                    #ROSData = genROS(radData,cells)
                    if self.ROSType == "Complex ROS":
                        ROSData = genROS(radData, cells)
                    if self.ROSType == "Basic ROS":
                        ROSData = genROSOld(radData, cells)

            
            # initialize list of cells that have moved
            movedCells = []
            # for every existing cell, determine whether a cell moves. If it does, write it to the list
            for c in cells:
                
                initPos = c.position
                initPos = [initPos[0],initPos[1],initPos[2]]
                movedCell = c.brownianMove(T,self.N,g)
                newPos = movedCell.position
                # if cell has moved
                if initPos != newPos and newPos != -1:
                    # document new position in simulation space, and assign old position to empty
                    T[initPos[0],initPos[1],initPos[2]] = 0
                    T[newPos[0],newPos[1],newPos[2]] = c.health
                    
                    movedCells.append(c)

            # initialize list of new cells
            newCells = []
            # for every existing cell, determine whether a cell replicates. If it does, write the new cell to the list
            for c in cells:
                
                health = c.health
                #position = c.position
                UUID = c.UUID
                
                # cell replication
                if health == 1:
                    # cellRepl returns either new cell, or same cell if saturation conditions occur
                    newCell = c.cellRepl(T,self.N,g)
                    newCellPos = newCell.position
                    newCellUUID = newCell.UUID
                    # if newCell the same as old cell, then saturation conditions occurred, and no replication took place
                    if newCellUUID != UUID and newCellPos != -1:
                        # only document new cell if old cell replicated    
                        # if new cell is avaialble, assign position as filled                
                        T[newCellPos[0],newCellPos[1],newCellPos[2]] = 1
                        newCells.append(newCell)
            
            
            # if radiation traversal has occured
            if (self.radType == "150 MeV Proton" and g == self.radGen and self.Gy != 0) or (self.radType == "Deep Space") or (self.radType == "GCRSim" and g == self.radGen) or (self.radType == "Gamma" and g >= self.radGen):
                # initialize list of cells affected by ion/electron energy depositions
                dirRadCells = []
                for c in cells:
                    health = c.health
                    if self.cellType == "wt":
                        radCell = c.cellRad(g,self.radGen,radData,self.radType)
                    elif self.cellType == "rad51":
                        radCell = c.cellRad_rad51(g,self.radGen,radData,self.radType)
                    if type(radCell) == Cell:
                        newHealth = radCell.health
                        if health != newHealth:
                            radCellPos = radCell.position
                            T[radCellPos[0],radCellPos[1],radCellPos[2]] = newHealth
                            dirRadCells.append(radCell)
                            ######################################################################
            # if ROS generation has occured (post-radiation)
            if (self.radType == "150 MeV Proton" and g >= self.radGen and self.Gy != 0) or (self.radType == "Deep Space") or (self.radType == "GCRSim" and g >= self.radGen) or (self.radType == "Gamma" and g >= self.radGen):
                # initialize list of cells affected by ROS
                ROSCells = []
                for c in cells:
                    health = c.health
                    if self.cellType == "wt":
                        ROSCell = c.cellROS(g,self.radGen,ROSData)
                    elif self.cellType == "rad51":
                        ROSCell = c.cellROS_rad51(g,self.radGen,ROSData)
                    newHealth = ROSCell.health
                    if health != newHealth:
                        ROSCellPos = ROSCell.position
                        T[ROSCellPos[0],ROSCellPos[1],ROSCellPos[2]] = newHealth
                        ROSCells.append(ROSCell)
            # if radiation has occured and cell type is NOT rad51 (cellType = wild type)
            if (self.radType == "150 MeV Proton" and g > self.radGen and self.Gy != 0 and self.cellType != "rad51") or (self.radType == "Deep Space" and self.cellType != "rad51") or (self.radType == "GCRSim" and g > self.radGen and self.cellType != "rad51") or (self.radType == "Gamma" and g >= self.radGen and self.cellType != "rad51"):
                # initialize list of cells that have undergone repair mechanisms
                repairedCells = []
                for c in cells:
                    health = c.health
                    if health == 2:
                        repairedCell = c.cellRepair(g)
                        newHealth = repairedCell.health
                        repairedCellPos = repairedCell.position
                        T[repairedCellPos[0],repairedCellPos[1],repairedCellPos[2]] = newHealth
                        repairedCells.append(repairedCell)
                        
            # documenting cell movement in the cells list
            for c in movedCells:
                cUUID = c.UUID
                newPos = c.position
                for c2 in cells:
                    c2UUID = c2.UUID
                    if cUUID == c2UUID:
                        c2.position = newPos
                    
            # documenting cell replication in the cells list
            cells.extend(newCells)
            
            # documenting cell damage
            # if radiation has occurred
            if (self.radType == "150 MeV Proton" and g >= self.radGen and self.Gy != 0) or self.radType == "Deep Space" or (self.radType == "GCRSim" and g >= self.radGen) or (self.radType == "Gamma" and g >= self.radGen):
                # for every cell damaged by ion or electron energy depositions
                for c in dirRadCells:
                    # get information about damaged cell
                    cUUID = c.UUID
                    newHealth = c.health
                    newSSBs = c.numSSBs
                    newDSBs = c.numDSBs
                    # find cell in cell list that matches damaged cell ID
                    for c2 in cells:
                        c2UUID = c2.UUID
                        if cUUID == c2UUID:
                            # adjust information about cell in cell list to reflect damage
                            c2.health = newHealth
                            c2.numSSBs = newSSBs
                            c2.numDSBs = newDSBs
                # for every cell damaged by ROS
                for c in ROSCells:
                    # get information about damaged cell
                    cUUID = c.UUID
                    newHealth = c.health
                    newSSBs = c.numSSBs
                    # find cell in cell list that matches damaged cell ID
                    for c2 in cells:
                        c2UUID = c2.UUID
                        if cUUID == c2UUID:
                            # adjust information about cell in cell list to reflect damage
                            c2.health = newHealth
                            c2.numSSBs = newSSBs
            
            #documenting cell repair
            # if cell can repair (not rad51), and radiation has occured
            if (self.radType == "150 MeV Proton" and g > self.radGen and self.Gy != 0 and self.cellType != "rad51") or (self.radType == "Deep Space" and self.cellType != "rad51") or (self.radType == "GCRSim" and g > self.radGen and self.cellType != "rad51") or (self.radType == "Gamma" and g >= self.radGen and self.cellType != "rad51"):
                # for every cell that has undergone repair
                for c in repairedCells:
                    # get information about repaired cell
                    cUUID = c.UUID
                    newHealth = c.health
                    newSSBs = c.numSSBs
                    newDSBs = c.numDSBs
                    # find cell in cell list that matches repaired cell ID
                    for c2 in cells:
                        c2UUID = c2.UUID
                        if cUUID == c2UUID:
                            # adjust information about cell in cell list to reflect repair
                            c2.health = newHealth
                            c2.numSSBs = newSSBs
                            c2.numDSBs = newDSBs
                            
            # adjust data with new generational data
            # column array to denote that new data entries are at the current generation
            genArr = np.ones([len(cells),1],dtype=int)*g
            # get cell information to store in data
            # initialize cellsHealth and cellsPos with placeholders
            cellsHealth = [0]
            cellsPos = [0,0,0]
            # for each cell, get all the associated information
            for c in cells:
                currPos = [c.position]
                currHealth = [c.health]
                # record all cell positions and healths in a list, with each cell being a new row
                cellsPos = np.vstack([cellsPos,currPos])
                cellsHealth = np.vstack([cellsHealth, currHealth])
            #remove placeholder values
            cellsPos = np.delete(cellsPos,(0),axis = 0)
            cellsHealth = cellsHealth[1:]
            # compile cell information with genArr
            newData = np.hstack([genArr,cellsPos,cellsHealth])
            
            # compile new generation data with the previous data
            data = np.vstack([data,newData])
        ######################################## Random decay, lifetime ROS for complex model ################################
            if self.ROSType == "Complex ROS":
                if g > self.radGen:
                    ROSDatak  , ROSData_decayed = train_test_split(ROSData, train_size = 0.5)
                    # half life 1 gen = .5, half life 2 gen = .707, half life 3 gen = .7937, 20 min half life = .125
                    ROSData = ROSDatak

            
            self.label_2.setText("Calculations complete. Plotting and writing data.")
            self.progressBar.setValue(current_value + 5)

            # for each simulation type, write the data to a text file titled by the radType
            # for each simulation type, plot the data as 1 figure/generation
    
            fig, ax = plt.subplots()

            if self.radType == "150 MeV Proton":
                datName = str(self.radAmount)+'Gy'
                dat_path = currResult_path + datName + ".txt"
                if self.fileWritten: 
                    np.savetxt(dat_path,data,delimiter = ',')
                    # if ROSData != 0: for 0 Gy
                    cellPlot(data, self.gen, radData,ROSData,self.radGen,self.N,plots_path)
                if self.display:
                    cellPlot(data, self.gen, radData,ROSData,self.radGen,self.N,ax)

            elif self.radType == "Deep Space":
                datName = 'deepSpace'
                dat_path = currResult_path + datName + ".txt"
                if self.fileWritten: 
                    np.savetxt(dat_path,data,delimiter = ',')
                    cellPlot(data, self.gen, radData,ROSData,self.radGen,self.N,plots_path)
                if self.display:
                    cellPlot(data, self.gen, radData,ROSData,self.radGen,self.N,ax)
                
            elif self.radType == "GCRSim":
                datName = 'GCRSim'
                dat_path = currResult_path + datName + ".txt"
                if self.fileWritten: 
                    np.savetxt(dat_path,data,delimiter = ',')
                    cellPlot(data, self.gen, radData,ROSData,self.radGen,self.N,plots_path)
                if self.display:
                    cellPlot(data, self.gen, radData,ROSData,self.radGen,self.N,ax)

            elif self.radType == "Gamma":
                datName = 'Gamma'
                dat_path = currResult_path + datName + ".txt"
                if self.fileWritten: 
                    np.savetxt(dat_path,data,delimiter = ',')
                    cellPlot(data, self.gen, radData,ROSData,self.radGen,self.N,plots_path)
                if self.display:
                    cellPlot(data, self.gen, radData,ROSData,self.radGen,self.N,ax)

            self.progressBar.setValue(100)
            self.label_17.setText("Plots and data written.")

            # Call the cellPlot function to generate the plot
            self.label.plt.show()

            # print("time elapsed: {:.2f}s".format(time.time() - start_time))

    
    """self.value = self.progressBar.value()
 
    if self.value == 100:
        self.pushbutton_8.setEnabled(True)"""

    def resultsScreen(self):
        self.stackedWidget.setCurrentIndex(2)
        self.elapsed_time = self.time.time() - self.start_time

        self.label_11.setLabelText("Time elasped: ", self.elasped_time)
        self.label_12.setLabelText("Runs: ", self.simAmt)
        self.label_13.setLabelText("Plots: ", self.plotCount)
        self.label_14.setLabelText("Simulations: ", self.simCount)
        
        if self.doData:
            self.label_7.setLabelText("Data written to Results folder.")
        if self.doPlot:
            if self.display == False:
                self.label_7.setLabelText("Plot(s) written to Results folder.")
            else:
                self.ui.label_4.setAlignment(Qt.AlignCenter)
                self.ui.label_4.setScaledContents(True)
        if self.doSim:
            if self.display == False:
                self.label_9.setLabelText("Simulations(s) written to Results folder.")

        if self.ui.pushButton_8.clicked:
            QApplication(app.exit)
            

if __name__ == "__main__":
    app = QApplication([])
    widget = Widget()
    widget.show()
    sys.exit(app.exec())
