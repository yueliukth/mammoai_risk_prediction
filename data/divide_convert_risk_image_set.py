import pandas as pd
import os
import convert_crop
import time
import sys

# data paths
inPath = '.'
outPath = '.'

# read the CSV file for mammoai data
df = pd.read_csv('.', delimiter=';', dtype={'sourcefile':str})

# make the train/val/test directories
trainPath = outPath + "train/"
valPath = outPath + "val/"
testPath = outPath + "test/"
if not os.path.exists(trainPath):
    os.makedirs(trainPath)
if not os.path.exists(valPath):
    os.makedirs(valPath)
if not os.path.exists(testPath):
    os.makedirs(testPath)

numStudies = df.shape[0]
sumRisk = 0
sumNoRisk = 0

i = 0
while i < numStudies:
	start = time.time()

	row = df.loc[i,:]
	
	if row['dicom_manufacturer'] != 'Siemens-Elema AB'\
	and not ((row['dicom_imagelaterality'] == row['x_cancer_laterality']) and (row['x_priorimage']==0))\
	and not ((row['x_implant'] == 1) and ("id" not in row['dicom_viewposition'].lower())):
		if row['x_dataset'] == 'training':
			outPath = trainPath
		elif row['x_dataset'] == 'validation':
			outPath = valPath 
		elif row['x_dataset'] == 'test':
			outPath = testPath

		print "image #" + str(i) + " " + row['studyPersonID'] + " " + outPath
	
		sourcefile = row['sourcefile'][2:]
		sourcefileSplit = sourcefile.split("\\")
		sourcefileAdjusted = '/'.join(sourcefileSplit)
		imagePath = inPath + sourcefileAdjusted
		imageBasename = os.path.basename(imagePath)

		convert_crop.process(imagePath, outPath+imageBasename[:-3]+'png', row['dicom_imagelaterality'].lower() == 'left')
		
		sumRisk += row['outcome_risk']
        sumNoRisk += 1-row['outcome_risk']

		print "Processing one image took: " + str(time.time() - start)
		sys.stdout.flush()

	i += 1

print "#risk=" + str(sumRisk) + "    #no_risk=" + str(sumNoRisk)
