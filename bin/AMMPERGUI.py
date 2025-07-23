# This Python file uses utf-8 encoding.

"""
Graphical User Interface for AMMPER v2.0

Created by Madeline Marous, in coordination with original code created by Amrita Singh and edited by Daniel Palacios.
Review README and Credits for more information.

"""
# GUI modules
import sys
from time import sleep
import subprocess
import random

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from vgui_form import Ui_Widget # AMMPER interface 
from movieMaker import movie_maker as mm

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
import matplotlib.pyplot as plt
start_time = time.time()

class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)

        self.ui.stackedWidget.setCurrentIndex(0)
        self.ui.label_33.setStyleSheet("color: white;")
        self.ui.label_34.setStyleSheet("color: white;")

        # Confirming connection from UI layout.
        self.stackedWidget = self.ui.stackedWidget

        self.radioButton = self.ui.radioButton
        self.radioButton_2 = self.ui.radioButton_2
        self.radioButton_3 = self.ui.radioButton_3
        self.radioButton_4 = self.ui.radioButton_4
        
        self.radioButton_5 = self.ui.radioButton_5
        self.radioButton_6 = self.ui.radioButton_6

        self.radioButton_7 = self.ui.radioButton_7
        self.radioButton_8 = self.ui.radioButton_8

        self.horizontalSlider = self.ui.horizontalSlider
        
        self.progressBar = self.ui.progressBar

        self.plainTextEdit = self.ui.plainTextEdit

        self.label = self.ui.label
        self.label_2 = self.ui.label_2
        self.label_3 = self.ui.label_3
        self.label_4 = self.ui.label_4
        self.label_5 = self.ui.label_5
        self.label_6 = self.ui.label_6
        self.label_7 = self.ui.label_7

        self.ui.checkBox.setEnabled(False)
        self.checkBox_2 = self.ui.checkBox_2
        self.checkBox_2.setChecked(True)

        # self.path = self.plainTextEdit.toPlainText()

        self.pushButton_4 = self.ui.pushButton_4

        # Connecting GUI framework to AMMPER code.
        self.ui.pushButton.clicked.connect(self.pushButton_clicked) # Launch GUI
        self.ui.pushButton_2.clicked.connect(self.pushButton_2_clicked) # Launch CLI
        self.ui.pushButton_3.clicked.connect(self.pushButton_3_clicked) # Credits
        self.ui.pushButton_4.clicked.connect(self.pushButton_4_clicked) # Set Up Simulation
        self.ui.pushButton_4.clicked.connect(self.simSetup) # Run Simulation
        self.ui.pushButton_5.clicked.connect(self.pushButton_5_clicked) # Exit
        self.ui.pushButton_6.clicked.connect(self.goBack) # Backwards Credits
        self.ui.pushButton_7.clicked.connect(self.visualization) # Visualization
        self.ui.pushButton_8.clicked.connect(self.pushButton_5_clicked) # Exit

        # Connecting radio buttons.
        self.radioButton.toggled.connect(self.onRadioButtonClicked)
        self.radioButton_2.toggled.connect(self.onRadioButtonClicked)
        self.radioButton_3.toggled.connect(self.onRadioButtonClicked)
        self.radioButton_4.toggled.connect(self.onRadioButtonClicked)
        
        self.radioButton_5.toggled.connect(self.onRadioButtonClicked2)
        self.radioButton_6.toggled.connect(self.onRadioButtonClicked2)

        self.radioButton_7.toggled.connect(self.onRadioButtonClicked3)
        self.radioButton_8.toggled.connect(self.onRadioButtonClicked3)

        self.ui.checkBox.stateChanged.connect(self.fileExport)
        self.ui.checkBox_2.stateChanged.connect(self.fileExport)

        self.horizontalSlider.valueChanged.connect(self.Slider)
        self.slider = self.horizontalSlider.value()

        # Initializing variables.

        self.radAmount = 0.0
        self.cellType = "" 
        self.radType = ""  
        self.N = 0 
        self.gen = 0
        self.ROSType = ""
        self.Gy = float(0)
        self.simDescription = ""
        self.sliderOn = True
        # Starting boolean
        self.display = True
        self.fileWritten = False
        self.pushButton_4.setEnabled(False)
        self.dirRadCells = []

    def pushButton_clicked(self):
        self.stackedWidget.setCurrentIndex(1)

    def pushButton_2_clicked(self):
        QApplication.exit()
        subprocess.call(["python", "AMMPERCLI.py"]) # Launch CLI

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

    def pushButton_5_clicked(self):
        QApplication.exit()

    def goBack(self):
        self.stackedWidget.setCurrentIndex(0)

    def onRadioButtonClicked(self):
        if self.radioButton.isChecked():
            self.sliderOn = True
            self.Gy = float(self.radAmount)
            self.radType = "150 MeV Proton"
            self.horizontalSlider.setEnabled(self.sliderOn)
            self.gen = 15
            self.radGen = 2
            self.N = 64

            if self.Gy == 0:
                self.radData = np.zeros([1,6],dtype = float)
                self.ROSData = np.zeros([1,6],dtype = float)

        if self.radioButton_2.isChecked():
            self.sliderOn = False
            self.radType = "GCRSim"
            self.horizontalSlider.setValue(1)
            self.ui.label_11.setText(str(0.5))
            self.horizontalSlider.setEnabled(self.sliderOn)
            self.gen = 15
            self.radGen = 2
            self.N = 64
            self.radData = np.zeros([1,6],dtype = float)
            self.ROSData = np.zeros([1,6],dtype = float)

        if self.radioButton_3.isChecked():
            self.sliderOn = False
            self.radType = "Deep Space"
            self.horizontalSlider.setValue(1)
            self.radAmount = 0
            self.ui.label_11.setText(str(0))
            self.horizontalSlider.setEnabled(self.sliderOn)
            self.gen = 15
            self.N = 300
            self.radGen = 0
            self.Gy = 0
            self.radData = np.zeros([1,6],dtype = float)
            self.ROSData = np.zeros([1,6],dtype = float)

        if self.radioButton_4.isChecked():
            self.sliderOn = False
            self.radType = "Gamma"
            self.horizontalSlider.setValue(1)
            self.ui.label_11.setText(str(0))
            self.horizontalSlider.setEnabled(self.sliderOn)
            self.gen = 15
            self.radGen = 10
            self.N = 64
            self.radData = np.zeros([1,6],dtype = float)
            self.ROSData = np.zeros([1,6],dtype = float)

        print(self.radType, self.sliderOn)

    def Slider(self):
        if self.sliderOn:
            self.slider = self.horizontalSlider.value()
            if self.slider == 1:
                self.radAmount = 0
            elif self.slider == 2:
                self.radAmount = 2.5
            elif self.slider == 3:
                self.radAmount = 5
            elif self.slider == 4:
                self.radAmount = 10
            elif self.slider == 5:
                self.radAmount = 20
            elif self.slider == 6:
                self.radAmount = 30
        else:
            if self.radType == "GCRSim":
                self.radAmount = 0.5

            if self.radType == "Deep Space" or self.radType == "Gamma":
                self.radAmount = 0

        self.Gy = float(self.radAmount)
        self.ui.label_11.setText(str(self.radAmount))

    def onRadioButtonClicked2(self):
        if self.radioButton_5.isChecked():
            self.cellType = "wt"
        elif self.radioButton_6.isChecked():
            self.cellType = "rad51"

    def onRadioButtonClicked3(self):
        if self.radioButton_7.isChecked():
            self.ROSType = "Basic ROS"
        elif self.radioButton_8.isChecked():
            self.ROSType = "Complex ROS"
        self.pushButton_4.setEnabled(True)

    def fileExport(self):
        if self.checkBox.isChecked():
            self.path = self.plainTextEdit.toPlainText()
            self.fileWritten = True
            try:
                with open(self.path, 'r'):
                    self.label_7.setText("Valid path.")
            except:
                self.label_7.setText("Specific exportation unavailable in pilot testing. Thank you for your patience.")
                self.label_7.setStyleSheet("color: red;")
        if self.checkBox_2.isChecked():
            self.display = False

    def simSetup(self):
        self.simDescription = "Cell Type: " + self.cellType + "\nRad Type: " + self.radType + "\nSim Dim: " +  str(self.N) + "microns\nNumGen: " + str(self.gen) + "ROS model: " + str(self.ROSType)

    # results folder name with the time that the simulation completed
        self.resultsName = time.strftime('%m-%d-%y_%H-%M') + "/"
        # determine path that all results will be written to
        resultsFolder = r"Results/"
        #currPath = os.path.dirname("AMMPER")
        allResults_path = os.path.join(resultsFolder)
        self.currResult_path = os.path.join(allResults_path,self.resultsName)
        plots_path = os.path.join(self.currResult_path,r"Plots/")

    # if any of the folders do not exist, create them
        if not os.path.isdir(resultsFolder):
            os.makedirs(resultsFolder)
        if not os.path.isdir(self.currResult_path):
            os.makedirs(self.currResult_path)
        if not os.path.isdir(plots_path):
            os.makedirs(plots_path)

        # write description to file
        np.savetxt(self.currResult_path+'simDescription.txt',[self.simDescription],fmt='%s')

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

        self.ui.label_10.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ui.label_10.setScaledContents(True)

    #placeholder initialization
        if self.radType == "Deep Space":
            self.radData = np.zeros([1,7],dtype = float)
            self.ROSData = np.zeros([1,7],dtype = float)

        self.fact_list = ["AMMPER incorporates data from BioSentinel, the first biological CubeSat to fly beyond Low Earth Orbit (LEO).",
                          "AMMPER models after yeast (Saccharomyces cerevisiae) cells because they are eukaryotic, therefore similar in biology to human cells.", 
                          "AMMPER was first created by Amrita Singh in 2021.", "AMMPER is currently available for download under an open-source agreement.", 
                          "AMMPER is the first microbial model (to our knowledge) to study the effects of deep-space radiation on individual cells."]
        self.random_item = random.choice(self.fact_list)
        self.label_3.setText("Did you know: " + self.random_item)

        self.label_2.setText("Simulation beginning.")
        current_value = 0
        for g in range(1,self.gen + 1):
            self.label_2.setText("Generation " + str(g))
            current_value = self.progressBar.value() 
            self.progressBar.setValue(current_value + 5)
            sleep(1)
            pathX = "/Results/" + self.resultsName + "/Plots/fig" + str(g) + "1.png"
            pixmapgX = QPixmap(pathX)
            self.ui.label_5.setPixmap(pixmapgX)

            if self.radType == "Gamma":
                if g == self.radGen:

                    dose = 1
                    # radData = np.zeros([1, 6], dtype=float)
                    # Dose input, radGenE stop point for gamma radiation.
                    self.radData = GammaRadGen(dose)
                    # radData = np.delete(radData, (0), axis=0)

                    if self.ROSType == "Complex ROS":
                        self.ROSData = genROS(self.radData, cells)
                    if self.ROSType == "Basic ROS":
                        self.ROSData = genROSOld(self.radData, cells)

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
                        self.radData = np.zeros([1,6],dtype = float)
                        # ROSData = np.zeros([1,6],dtype = float)

                        for track in trackChoice:
                            trackNum = track
                            # creates a traverse for every track in trackChoice
                            radData_trans = genTraverse_groundTesting(self.N,protonEnergy,trackNum,energyThreshold,self.radType)
                            # compile all energy depositions from individual tracks together
                            self.radData = np.vstack([self.radData,radData_trans])

                        #remove placeholder of 0s from the beginning of radData
                        self.radData = np.delete(self.radData,(0),axis = 0)

                        # direct energy results in ROS generation - use energy depositions to calculate ROS species
                        if self.ROSType == "Complex ROS":
                            self.ROSData = genROS(self.radData,cells)
                        if self.ROSType == "Basic ROS":
                            self.ROSData = genROSOld(self.radData,cells)

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
                            self.radData = np.vstack([self.radData,radData_trans])

                            # compile ROSData with the generation indicator
                            ROSData_new = np.hstack((ROSData_new,genArr))
                            #compile ROSData with all ROSData
                            self.ROSData = np.vstack([self.ROSData,ROSData_new])

            elif self.radType == "GCRSim":
                if g == self.radGen:
                    # placeholder initialization
                    self.radData = np.zeros([1,6],dtype = float)
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
                        self.radData = np.vstack([self.radData,radData_trans])

                        #remove placeholder from beginning
                        self.radData = np.delete(self.radData,(0),axis = 0)
                    # generate ROS data from all traversal energy depositions
                    #ROSData = genROS(radData,cells)
                    if self.ROSType == "Complex ROS":
                        self.ROSData = genROS(self.radData, cells)
                    if self.ROSType == "Basic ROS":
                        self.ROSData = genROSOld(self.radData, cells)


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
                self.dirRadCells = []
                for c in cells:
                    health = c.health
                    if self.cellType == "wt":
                        radCell = c.cellRad(g,self.radGen,self.radData,self.radType)
                    elif self.cellType == "rad51":
                        radCell = c.cellRad_rad51(g,self.radGen,self.radData,self.radType)
                    if type(radCell) == Cell:
                        newHealth = radCell.health
                        if health != newHealth:
                            radCellPos = radCell.position
                            T[radCellPos[0],radCellPos[1],radCellPos[2]] = newHealth
                            self.dirRadCells.append(radCell)
                            ######################################################################
            # if ROS generation has occured (post-radiation)
            if (self.radType == "150 MeV Proton" and g >= self.radGen and self.Gy != 0) or (self.radType == "Deep Space") or (self.radType == "GCRSim" and g >= self.radGen) or (self.radType == "Gamma" and g >= self.radGen):
                # initialize list of cells affected by ROS
                ROSCells = []
                for c in cells:
                    health = c.health
                    if self.cellType == "wt":
                        ROSCell = c.cellROS(g,self.radGen,self.ROSData)
                    elif self.cellType == "rad51":
                        ROSCell = c.cellROS_rad51(g,self.radGen,self.ROSData)
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

                for c in self.dirRadCells:
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

        
        self.label_2.setText("Complete. Stand by.")
        self.progressBar.setValue(current_value + 5)

        # for each simulation type, write the data to a text file titled by the radType
        # for each simulation type, plot the data as 1 figure/generation
  
        #fig, ax = plt.subplots()

        if self.radType == "150 MeV Proton":
            datName = str(self.radAmount)+'Gy'
            dat_path = self.currResult_path + datName + ".txt"
            if self.fileWritten: 
                np.savetxt(dat_path,data,delimiter = ',')
                # if ROSData != 0: for 0 Gy
                cellPlot(data, self.gen, self.radData,self.ROSData,self.radGen,self.N,plots_path)
            if self.display:
                cellPlot(data, self.gen, self.radData,str(self.ROSData),str(self.radGen),self.N, plots_path)

        elif self.radType == "Deep Space":
            datName = 'deepSpace'
            dat_path = self.currResult_path + datName + ".txt"
            if self.fileWritten: 
                np.savetxt(dat_path,data,delimiter = ',')
                cellPlot(data, self.gen, self.radData,self.ROSData,self.radGen,self.N,plots_path)
            if self.display:
                cellPlot(data, self.gen, self.radData,self.ROSData,self.radGen,self.N,plots_path)
            
        elif self.radType == "GCRSim":
            datName = 'GCRSim'
            dat_path = self.currResult_path + datName + ".txt"
            if self.fileWritten: 
                np.savetxt(dat_path,data,delimiter = ',')
                cellPlot(data, self.gen, self.radData,self.ROSData,self.radGen,self.N,plots_path)
            if self.display:
                cellPlot(data, self.gen, self.radData,self.ROSData,self.radGen,self.N,plots_path)

        elif self.radType == "Gamma":
            datName = 'Gamma'
            dat_path = self.currResult_path + datName + ".txt"
            if self.fileWritten: 
                np.savetxt(dat_path,data,delimiter = ',')
                cellPlot(data, self.gen, self.radData,self.ROSData,self.radGen,self.N,plots_path)
            if self.display:
                cellPlot(data, self.gen, self.radData,self.ROSData,self.radGen,self.N,plots_path)

        self.progressBar.setValue(100)
        self.label_2.setText("Plots and data written.")
        self.ui.label.setText("Time elapsed: \n{:.2f}s".format(time.time() - start_time))

        sleep(1)

        #gen1
        self.path1 = "Results/" + self.resultsName + "/Plots/fig1.png"
        self.pixmapg1 = QPixmap(self.path1)

        #gen2
        self.path2 = "Results/" + self.resultsName + "/Plots/fig2.png"
        self.pixmapg2 = QPixmap(self.path2)

        #gen3
        self.path3 = "Results/" + self.resultsName + "/Plots/fig3.png"
        self.pixmapg3 = QPixmap(self.path3)
        
        #gen4
        self.path4 = "Results/" + self.resultsName + "/Plots/fig4.png"
        self.pixmapg4 = QPixmap(self.path4)

        #gen5
        self.path5 = "Results/" + self.resultsName + "/Plots/fig5.png"
        self.pixmapg5 = QPixmap(self.path5)

        #gen6
        self.path6 = "Results/" + self.resultsName + "/Plots/fig6.png"
        self.pixmapg6 = QPixmap(self.path6)

        #gen7
        self.path7 = "Results/" + self.resultsName + "/Plots/fig7.png"
        self.pixmapg7 = QPixmap(self.path7)

        #gen8
        self.path8 = "Results/" + self.resultsName + "/Plots/fig8.png"
        self.pixmapg8 = QPixmap(self.path8)

        #gen9
        self.path9 = "Results/" + self.resultsName + "/Plots/fig9.png"
        self.pixmapg9 = QPixmap(self.path9)

        #gen10
        self.path10 = "Results/" + self.resultsName + "/Plots/fig10.png"
        self.pixmapg10 = QPixmap(self.path10)
        
        #gen11
        self.path11 = "Results/" + self.resultsName + "/Plots/fig11.png"
        self.pixmapg11 = QPixmap(self.path11)

        #gen12
        self.path12 = "Results/" + self.resultsName + "/Plots/fig12.png"
        self.pixmapg12 = QPixmap(self.path12)

        #gen13
        self.path13 = "Results/" + self.resultsName + "/Plots/fig13.png"
        self.pixmapg13 = QPixmap(self.path13)

        #gen14
        self.path14 = "Results/" + self.resultsName + "/Plots/fig14.png"
        self.pixmapg14 = QPixmap(self.path14)

        #gen15
        self.path15 = "Results/" + self.resultsName + "/Plots/fig15.png"
        self.pixmapg15 = QPixmap(self.path15)

        self.stackedWidget.setCurrentIndex(3)

        self.ui.label_4.setPixmap(self.pixmapg1)
        self.ui.label_19.setPixmap(self.pixmapg5)
        self.ui.label_12.setPixmap(self.pixmapg9)
        self.ui.label_5.setPixmap(self.pixmapg12)
        self.ui.label_13.setPixmap(self.pixmapg15)

        # Setting label properties
        self.ui.label_4.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ui.label_4.setScaledContents(True)
        self.ui.label_19.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ui.label_19.setScaledContents(True) 
        self.ui.label_12.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ui.label_12.setScaledContents(True) 
        self.ui.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ui.label.setScaledContents(True) 
        self.ui.label_5.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ui.label_5.setScaledContents(True) 
        self.ui.label_13.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ui.label_13.setScaledContents(True)

    def visualization(self):
        self.stackedWidget.setCurrentIndex(5)
        self.ui.label_35.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ui.label_35.setScaledContents(True)
        self.ui.label_35.setScaledContents(True)
        mm.movieMaker("Results/", self.path1, self.path2, self.path3, self.path4, self.path5, self.path6, self.path7, self.path8, self.path9, self.path10, self.path11, self.path12, self.path13, self.path14, self.path15)
        videopath = QPixmap("Results/visualization.mp4")
        self.ui.label_35.setPixmap(videopath)
       

# Widget initialization. 

if __name__ == "__main__":
    app = QApplication([])
    widget = Widget()
    widget.show()
    sys.exit(app.exec())
    widget.simDescription()
