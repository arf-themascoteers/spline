import os

TEST = False
LUCAS_PATH = r"D:\Data\LUCAS\Lucas-2015"
DATASET = "data/dataset_s2.csv"
SPECTRA_DIR = os.path.join(LUCAS_PATH,"LUCAS2015_Soil_Spectra_EU28")
if os.path.exists(SPECTRA_DIR):
    SPECTRA_FILES = [os.path.join(SPECTRA_DIR, file) for file in os.listdir(SPECTRA_DIR)]
TOPSOIL_FILE = os.path.join(os.path.join(LUCAS_PATH, "LUCAS2015_topsoildata_20200323"),"LUCAS_Topsoil_2015_20200323.csv")
LIGHTNING = False

if TEST:
    SPECTRA_FILES = [os.path.join(SPECTRA_DIR, "spectra_ MT .csv")]
    DATASET = "data/dataset_min.csv"