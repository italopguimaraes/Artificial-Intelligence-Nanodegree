import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    Xlengths = test_set.get_all_Xlengths()
    for x in Xlengths:
        try:
            s={}
            for m in models:
                try:
                    if models[m] is not None:
                        p=models[m].score(Xlengths[x][0],Xlengths[x][1])
                        s[m]=p
                    else:
                        continue
                except ValueError:
                    continue
            probabilities.append(s)
        except ValueError:
            continue
            
    for p in probabilities:
        max_log=float('-inf')
        word=None
        for x in p:
            if p[x]>max_log:
                max_log=p[x]
                word=x
        guesses.append(word)
    return (probabilities,guesses)
            
    
            
            
    
        
        
