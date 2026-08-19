Intro about runs 

Code pipeline:

AMMPERruns_aB.py (or AMMPErruns_GAMMAFINAL.py) calls AMMPERBulk_aB.py (or AMMPERBulk_GAMMAFINAL.py) to run AMMPER n times at specific parameters. It is a simple for loop taking 
user inputs to system arguments and passing through them. 

Results are store in Results_Bulk_aB (or Results_Bulk_GAMMAFINAL) which correspond to AMMPER prediction
	Each subfolder correspond to specific set of parameters e.g. WT 2.5 Gy ion radiation within each subfolder we have AMMPER results which are read by following scripts

Then we run aB model which generates the plots comparing experimental vs predicted AMMPER data with aBFinalplots.py (or aBFinalplotsGAMMA.py)

Task that we need to do:
Take a look at biosentinel data folder, here it is the ion and gamma radiation experimental raw datasets. Note some preprocessing is due to call this data into our aB model. 
Modifying this might be necessary to avoid manually creating csv files e.g. AlamarblueRawdatarad5130GySTD.csv. 

We need to run AMMPER for each radiation dose, cell type and radiation type i.e. 

Doses: 0, 2.5, 5, 10, 20, 30 Gy 
Radiation types: Ion, Gamma
Cell types: WT, rad51

Note current pipeline only plots 0, 2.5 and 30 Gy for Ion (WT) and Gamma (rad51) 

Warnings:
1.- We need to repeat analysis radGen needs to be 2 for both ion and Gamma
2.- For gamma experimental data is slightly delayed we need to adjust prediction and experimental times. I think it starts at hour 4 this can be done in code or skipping the first 4 hours from raw data while feeding to model


Procedure:

1.- Re run pipeline with radGen = 2 instead of radGen = 12.
2.- Once we can re run pipeline, we just need to do it for the rest of doses, and cell types ... 


