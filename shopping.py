import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd


TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    #print (len(evidence))
    #print (len(labels))
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - DONE - Administrative, an integer
        - Administrative_Duration, a floating point number
        - DONE - Informational, an integer
        - Informational_Duration, a floating point number
        - DONE - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - DONE - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - DONE - OperatingSystems, an integer
        - DONE - Browser, an integer
        - DONE - Region, an integer
        - DONE - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence_list=[]
    labels_list=[]
    
    f = open(filename, 'r')
    #f.readline()  # skip the header
    lines = list(csv.DictReader(f))
    elemantslines= lines
    del elemantslines[-1]
    

    for line in lines:
        #convert to integers
        for field in ['Administrative', 'Informational', 'ProductRelated', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']:
            line[field] = int(line[field])
            # print (line[field])
                
        #convert to floating points
        for fields in ['Administrative_Duration','Informational_Duration','ProductRelated_Duration','BounceRates','ExitRates','PageValues','SpecialDay' ]:
            line[field] = float(line[field])
            # print (line[field])
        #convert months
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for i in range(len(months)):
            if line['Month'] == months[i]:
                line['Month'] = i
                break
    
        
        #convert visitor type
        for fields in ['VisitorType']:
            if line[fields]== "Returning_Visitor":
                line[fields]=1
            else:
                line[fields]=0
        
        #convert weekend
        for fields in ['Weekend']:
            if line[fields]== "TRUE":
                line[fields]=1
            else:
                line[fields]=0
                
        evidence_list.append(list(line.values())[:-1])
    
    
        for field in ['Revenue']:
            if line[field]== "TRUE":
                line[field]=1
                labels_list.append(1)
            else:
                labels_list.append(0)
                
        
    return evidence_list, labels_list             

       
        
#    data = np.loadtxt(f)
 #   x = data[:, :-1]  # select columns 1 through end
  #  y = data[:, -1] 
    
    
    '''
    
    df = pd.read_csv(filename, header = 0)
    original_headers = list(df.columns.values)
    df = df._get_numeric_data()
    numeric_headers = list(df.columns.values)
    numpy_array = df.as_matrix()
'''


    
   # print (x)
   # print (y)
    
    
    #raise NotImplementedError


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model
    


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    #labels = labels.astype(np.float64)
    #predictions = predictions.astype(np.float64)
    sensitivity = float(0)
    specificity =  float(0)
    FN = float(0)
    FP = float(0)
    TP = float(0)
    TN = float(0)
    
    '''
    for each l,p in (labels,predictions)
        check if the label matched the predictions 
            if both positive then TP+=1
            if both negative then TN+=1
        check if label dont match predictions
            if label is positive, then FP+=1
            if label is negative then FN+=1
    sensitivity = tp/(tp+fp)
    specificity = tn/(tn+fn)
    '''
    for label in labels:
        for pred in predictions:
            if label == pred:
                if pred==1:
                    TP+=1
                else:
                    TN+=1
            else:
                if label ==1:
                    FP+=1
                else:
                    FN+=1
    
    sensitivity= TP/(TP+FN)
    specificity= TN/(TN+FP)
    return(sensitivity,specificity)

        
    #raise NotImplementedError


if __name__ == "__main__":
    main()
