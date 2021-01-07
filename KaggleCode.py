from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import numpy as np 
import pandas
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
import copy 
from sklearn.ensemble import BaggingClassifier


def createOutput(aDF, results, fileName):
	newDF = copy.deepcopy(aDF)
	newDF['label'] = results
	newDF.to_csv(fileName, index=False)

def findAccuracy(y,yhat):
	newy = copy.deepcopy(y)
	newy = newy.reset_index(drop=True)
	total = len(newy)
	correct = 0 
	for i in range(len(newy)):
		if (newy[i] == yhat[i]):
			correct = correct + 1 

	return (float(correct)/total)

# Load dataset
trainDF = pandas.read_csv("train.csv")
testDF = pandas.read_csv("test.csv")

# Prepare dataframe for final output 
outputTreeTestDF = pandas.read_csv("test.csv")
idHeader = ['id']
outputTreeTestDF = outputTreeTestDF[idHeader]

# Adjust string values to numeric representation
trainDF.loc[trainDF['category'] == 'jam', 'category'] = 1 
trainDF.loc[trainDF['category'] == 'compo', 'category'] = 2 
testDF.loc[testDF['category'] == 'jam', 'category'] = 1 
testDF.loc[testDF['category'] == 'compo', 'category'] = 2 

# Features to keep with and without label column 

featuresKeep = ['category', 'num-comments', 'ratings-received', 'prev-games', 'fun-average', 
				'innovation-average', 'theme-average', 'graphics-average', 'audio-average', 'humor-average',
				'mood-average', 'fun-rank', 'innovation-rank', 'theme-rank', 'graphics-rank', 'audio-rank', 
				'humor-rank', 'mood-rank', 'label']

featuresKeepTest = ['category', 'num-comments', 'ratings-received', 'prev-games', 'fun-average', 
				'innovation-average', 'theme-average', 'graphics-average', 'audio-average', 'humor-average',
				'mood-average', 'fun-rank', 'innovation-rank', 'theme-rank', 'graphics-rank', 'audio-rank', 
				'humor-rank', 'mood-rank']

# Prepare train data for model input 
trainDF = trainDF[featuresKeep]
Xvalue = trainDF[featuresKeepTest]
Yvalue = trainDF['label']

# Prepare test data for prediction 
testDF = testDF[featuresKeepTest]
testValuesX = testDF.values

# Train test splits 
X_train, X_test, y_train, y_test = train_test_split(Xvalue, Yvalue, test_size = 0.2, random_state = 1)
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

#################################################################################################################
## Basic Decision Tree 
#################################################################################################################

# Basic Decision Tree
basicClassifier = DecisionTreeClassifier()

# Train Decision Tree Classifer
basicClassifier = basicClassifier.fit(X_train,y_train)

# Prediction of X_test
y_predBasic = basicClassifier.predict(X_test)
acc = findAccuracy(y_test, y_predBasic)
print("The accuracy for basicClassifier is : {}".format(acc))

#################################################################################################################
## Max Depth + Entropy Decision Tree 
#################################################################################################################

# Limit max depth to avoid overfitting and use info gain method 
secondClassifier = DecisionTreeClassifier(criterion="entropy", splitter='best', max_depth=13)

# Train Decision Tree Classifer
#secondClassifier = secondClassifier.fit(X_train,y_train)
secondClassifier = secondClassifier.fit(X_train,y_train)

# Prediction of X_test
y_predSecond = secondClassifier.predict(X_test)
acc = findAccuracy(y_test, y_predSecond)
print("The accuracy for Classifier with max depth + entropy is : {}".format(acc))

#################################################################################################################
## Max Depth + Entropy Decision Tree  + (BOOTSTRAPING)
#################################################################################################################

# Try to create multiple decision trees and take the mode of all predictions
# BaggingClassifier will create and execute based on number of inputted trees 

modelBag1 = BaggingClassifier(n_estimators = 1)
modelBag10 = BaggingClassifier(n_estimators = 10)
modelBag50 = BaggingClassifier(n_estimators = 50)
modelBag100 = BaggingClassifier(n_estimators = 100)
modelBag200 = BaggingClassifier(n_estimators = 200)
modelBag300 = BaggingClassifier(n_estimators = 300)

arrayBagModels = [modelBag1, modelBag10, modelBag50, modelBag100, modelBag200, modelBag300]
arrayBagModelsName = ['modelBag1', 'modelBag10', 'modelBag50', 'modelBag100', 'modelBag200', 'modelBag300']

# Prediction accuracy of each bagged model on X_test from initial train test split 
for i in range(len(arrayBagModels)):
	arrayBagModels[i].fit(X_train,y_train)
	y_predBagging = arrayBagModels[i].predict(X_test)
	acc = findAccuracy(y_test, y_predBagging)
	print("The accuracy for Classifier with Bagging {} is : {}".format(arrayBagModelsName[i],acc))


# Based on test bagging with 200 had the best results 
finalBagModel = BaggingClassifier(n_estimators = 200)

# Fit on entire train data set 
finalBagModel.fit(Xvalue, Yvalue)


#################################################################################################################
## (BOOTSTRAPING) V2 with tree max depth 13 
#################################################################################################################

# Try to create multiple decision trees and take the mode of all predictions
# BaggingClassifier will create and execute based on number of inputted trees 

maxDTree = DecisionTreeClassifier(criterion="entropy", splitter='best', max_depth=13)
modelBag100v2 = BaggingClassifier(maxDTree, n_estimators = 100, max_samples=0.8, random_state = 1)
modelBag100v2.fit(Xvalue, Yvalue)


y_predBaggingV2 = modelBag100v2.predict(X_test)
acc = findAccuracy(y_test, y_predBaggingV2)
print("The accuracy for Classifier with Bagging V2 is : {}".format(acc))


#################################################################################################################
## (BOOTSTRAPING) V3 with tree max depth 13 + 200 trees 
#################################################################################################################

# Try to create multiple decision trees and take the mode of all predictions
# BaggingClassifier will create and execute based on number of inputted trees 

maxDTree = DecisionTreeClassifier(criterion="entropy", splitter='best', max_depth=13)
modelBag100v3 = BaggingClassifier(maxDTree, n_estimators = 200, max_samples=0.8, random_state = 1)
modelBag100v3.fit(Xvalue, Yvalue)


y_predBaggingV3 = modelBag100v3.predict(X_test)
acc = findAccuracy(y_test, y_predBaggingV3)
print("The accuracy for Classifier with Bagging V3 is : {}".format(acc))


#################################################################################################################
## (BOOTSTRAPING) V4 with tree max depth 14 + 200 trees 
#################################################################################################################

# Try to create multiple decision trees and take the mode of all predictions
# BaggingClassifier will create and execute based on number of inputted trees 

maxDTree = DecisionTreeClassifier(criterion="entropy", splitter='best', max_depth=14)
modelBag100v4 = BaggingClassifier(maxDTree, n_estimators = 200, max_samples=0.8, random_state = 1)
modelBag100v4.fit(Xvalue, Yvalue)


y_predBaggingV4 = modelBag100v4.predict(X_test)
acc = findAccuracy(y_test, y_predBaggingV4)
print("The accuracy for Classifier with Bagging V4 is : {}".format(acc))

#################################################################################################################
## (BOOTSTRAPING) V5 with tree max depth 14 + 200 trees 
#################################################################################################################

# Try to create multiple decision trees and take the mode of all predictions
# BaggingClassifier will create and execute based on number of inputted trees 

maxDTree = DecisionTreeClassifier(criterion="entropy", splitter='best', max_depth=12)
modelBag100v5 = BaggingClassifier(maxDTree, n_estimators = 100, max_samples=0.8, random_state = 1)
modelBag100v5.fit(Xvalue, Yvalue)


y_predBaggingV5 = modelBag100v5.predict(X_test)
acc = findAccuracy(y_test, y_predBaggingV5)
print("The accuracy for Classifier with Bagging V5 is : {}".format(acc))

#################################################################################################################
## Output files when using a Tree as a classifier
#################################################################################################################

results1 = basicClassifier.predict(testValuesX)
results2 = secondClassifier.predict(testValuesX)
results3 = finalBagModel.predict(testValuesX)
results4 = modelBag100v2.predict(testValuesX)
results5 = modelBag100v3.predict(testValuesX)
results6 = modelBag100v4.predict(testValuesX)
results7 = modelBag100v5.predict(testValuesX)

createOutput(outputTreeTestDF, results1, "outputTestTreeBasic.csv")
createOutput(outputTreeTestDF, results2, "outputTestTreeMaxDepth.csv")
createOutput(outputTreeTestDF, results3, "outputTestTreeBagging.csv")
createOutput(outputTreeTestDF, results4, "outputTestTreeBaggingV2.csv")
createOutput(outputTreeTestDF, results5, "outputTestTreeBaggingV3.csv")
createOutput(outputTreeTestDF, results6, "outputTestTreeBaggingV4.csv")
createOutput(outputTreeTestDF, results7, "outputTestTreeBaggingV5.csv")

