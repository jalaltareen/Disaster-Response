import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pickle



def load_data(database_filepath):

    
    '''
    Function for loading the dataframe or table from database
    
    INPUT 
        database_filepath - Path of file for importing from database.     
    
    OUTPUT
        Returns the following variables:
        X - Returns the input features.  The message column from database
        Y - Returns the categories of the dataset.  Important and used for classification of Input variable X
        keys - Columns of the Y.
    
    '''
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM Messages", engine)
    X = df.message.values
    Y = df.iloc[:,5:]
    col_key = Y.keys()
    
    return X, Y, col_key

def tokenize(text):
    
    '''
    Process the text and convert that into token
    
    INPUT 
        text: Text to be processed   
    OUTPUT
        Returns a processed text variable that was tokenized, lower cased, stripped, and lemmatized
    
    '''
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens


def build_model(X_train,y_train):
    
    '''
    INPUT 
        X_Train: Training features for use by GridSearchCV
        y_train: Training labels for use by GridSearchCV
    OUTPUT
        Returns a pipeline model that has gone through tokenization, count vectorization, 
        TFIDTransofmration and created into a ML model
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {  
        'clf__estimator__min_samples_split': [2, 4],
        #'clf__estimator__max_features': ['log2', 'auto', 'sqrt', None],
        #'clf__estimator__criterion': ['gini', 'entropy'],
        #'clf__estimator__max_depth': [None, 25, 50, 100, 150, 200],
        #'tfidf__use_idf': (True, False),
        #'clf__estimator__n_estimators': [50, 60, 70],
    }
    
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
    cv.fit(X_train,y_train)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)

    # Print out the full classification report and accuracy score
    print(classification_report(Y_test, Y_pred, target_names=category_names))

    #global categories
    
    for col in range(Y_test.shape[1]):
        
        print('Accuracy of %25s: %.2f' %(category_names[col], accuracy_score(Y_test.iloc[:, col].values, Y_pred[:,col])))
        
        

def save_model(model, model_filepath):
    '''
    Saves the model to disk
    INPUT 
        model: The model to be saved
        model_filepath: Filepath for where the model is to be saved
    OUTPUT
        While there is no specific item that is returned to its calling method, this method will save the model as a pickle file.
    '''    
    temp_pickle = open(model_filepath, 'wb')
    pickle.dump(model, temp_pickle)
    temp_pickle.close()

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train, Y_train)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()