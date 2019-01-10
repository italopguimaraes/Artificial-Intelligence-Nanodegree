import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_model=None
        min_BIC=float('inf')
        num_max_iteration=5
        # TODO implement model selection based on BIC scores
        logL=None
        for n in range(self.min_n_components,self.max_n_components):
            try:
                #model = GaussianHMM(n_components=n,covariance_type="diag", n_iter=1000,random_state=self.random_state, verbose=False).fit(X,lengths)
                count_1=0
                while count_1<num_max_iteration:
                    model =self.base_model(n)
                    if model is not None:
                        logL = model.score(self.X, self.lengths)
                    else:
                        count_1+=1
                        continue
                    if model is not None and logL is not None:
                        break
                    count_1+=1
                if model is None or logL is None:
                    continue
                #calculates the BIC where the number of parameters, is the number of states, as I do not know the correct value of P I used N, I would like you to confirm this for me
                p=n**2+(2*n*len(self.X[0])-1)
                BIC = -2*logL+p*math.log(len(self.X[0]),10)
                #check if the new value calculated for BIC is less than the previous calculated value, in this case, this will be the new BIC value, and the best model will be the model that generated it, in case of some error returns the best model
                if BIC<min_BIC:
                    min_BIC=BIC
                    best_model=model
            except ValueError:
                continue
            
        return best_model
                


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_model=None
        max_DIC=float('-inf')
        num_max_iteration=5
        logP_Xi=None 
        #trains the model using as resources provided for the word in which one wishes to train the particular model, for the number of states varying between self.min_n_components and self.max_n_components,
        for n in range(self.min_n_components,self.max_n_components):
            try:
                count_1=0
                while count_1<num_max_iteration:
                    model=self.base_model(n)
                    if model is not None:
                        break
                    count_1+=1
                if model is None:
                    continue
                else:
                    count_1=0
                    while count_1<num_max_iteration:
                        logP_Xi = model.score(self.X, self.lengths)
                        if logP_Xi is not None:
                            break
                        count_1+=1
                    if logP_Xi is None:
                        continue
                    
                average=0
                count=0
                #calculates the average of the anti-probability terms, that is the average of the competing data
                for a in self.hwords:
                    print(a)
                    xx=0
                    while xx<num_max_iteration:
                        if a!=this_word:
                            logP_Xj = model.score(self.hwords[a][0], self.hwords[a][1])
                        if logP_Xj is not None:
                            break
                        xx+=1
                    if logP_Xj is None:
                        continue
                    else:
                        average+=logP_Xj
                        count+=1
                average=average/count
                #calculates DIC
                DIC=logP_Xi-average
                #verifies that the new calculated value for DIC is greater than the previously calculated value, in which case this will be the new DIC value, and this will be the best model
                if DIC>max_DIC:
                    max_DIC=DIC
                    best_model=model
            except ValueError:
                continue
        #print('logP_Xi=',logP_Xi)
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection using CV
        #creates the list Features with appropriate resources the combine_sequences
        Features=[]
        cont=0
        num_max_iteration=5
        for a in self.X:
            a=a.tolist()
            cont_group=0
            l=[]
            while cont_group<self.lengths[cont]:
                l.append(a)
                cont_group+=1
            Features.append(l)
            cont+=1
            if cont>=len(self.lengths):
                break
        split_method = KFold()
        best_model=None
        best_logL_average=float('-inf')
        logL=None
        if len(Features)<num_max_iteration:
            #trains the models for number of states varying between self.min_n_components and self.max_n_components, using as resources the previously created lists          
            for n in range(self.min_n_components,self.max_n_components):
                try:
                    kont_xx=0
                    while kont_xx<num_max_iteration:
                        model = self.base_model(n)
                        if model is not None:
                            break
                        kont_xx+=1
                    if model is None:
                        continue
                    average=0
                    count=0
                    #calculates the average log Likelihood for all samples for this model
                    while count<len(Features):
                        kont_xx=0
                        while kont_xx<num_max_iteration:
                            logL = model.score(Features[count],[len(Features[count])])
                            if logL is not None:
                                break
                            kont_xx+=1
                        if logL is None:
                            continue
                        else:
                            average+=logL
                            count+=1
                    average=average/count
                    #if the new calculated average is greater than the previous one, then this will be the new media, and the model that generated it will be the best model, there will be some excess returns the best model
                    if average>best_logL_average:
                        best_logL_average=logL
                        best_model=model  
                except ValueError:
                    continue
        else:
            #uses cross-validation to split resources where cv_train_idx will have the indices for each round
            for cv_train_idx, cv_test_idx in split_method.split(Features):
                train=cv_train_idx.tolist()
                #uses the combine_sequences function to concatenate the sequences to train the model
                X,lengths=combine_sequences(train,Features)
                #trains the models for number of states varying between self.min_n_components and self.max_n_components, using as resources the previously created lists
                for n in range(self.min_n_components,self.max_n_components):
                    try:
                        kont_xx=0
                        while kont_xx<num_max_iteration:
                            model = self.base_model(n)
                            if model is not None:
                                break
                            kont_xx+=1
                        if model is None:
                            continue
                        average=0
                        count=0
                        #calculates the average log Likelihood for all samples for this model
                        test=cv_test_idx.tolist()
                        for t in test:
                            kont_xx=0
                            while kont_xx<num_max_iteration:
                                logL = model.score(Features[t],[len(Features[t])])
                                if logL is not None:
                                    break
                                kont_xx+=1
                            if logL is None:
                                continue
                            else:
                                average+=logL
                                count+=1
                        average=average/count
                        #if the new calculated average is greater than the previous one, then this will be the new media, and the model that generated it will be the best model, there will be some excess returns the best model
                        if average>best_logL_average:
                            best_logL_average=logL
                            best_model=model  
                    except ValueError:
                        continue
        #print('logL=',best_logL_average)
        return best_model

          
