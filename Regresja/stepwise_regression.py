import pandas as pd
import statsmodels.api as sm

def foreward_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01):
    """ 
    Funkcja wykonuje selekcję forward zmiennych 
    na podstawie p-value z statsmodels.api.OLS
    Argumenty:
        X - tablica ze zmiennymi objaśniającymi (pandas.Dataframe)
        y - Zmienna w postaci listy ze zmienną objaśnianą (array)
        initial_list - lista zmiennych które ręcznie wprowadzamy
            do modelu (array z nazwami kolumn z tablicy X)
        threshold_in - dodaj zmienną jeśli jej p-value < threshold_in (decimal)
        verbose - Czy wyświetlić kolejność dodawania/usuwania zmiennych (boolean)
    Zwraca: lista wyselekcjonowanych zmiennych
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
        if not changed:
            break
    return included

def backward_selection(X, y, threshold_out = 0.05):
    """ 
    Funkcja wykonuje selekcję forward zmiennych 
    na podstawie p-value z statsmodels.api.OLS
    Argumenty:
        X - tablica ze zmiennymi objaśniającymi (pandas.Dataframe)
        y - Zmienna w postaci listy ze zmienną objaśnianą (array)
        threshold_out - algorytm usuwa zmienną jeśli jej p-value > threshold_in (decimal)
        verbose - Czy wyświetlić kolejność dodawania/usuwania zmiennych (boolean)
    Zwraca: lista wyselekcjonowanych zmiennych
    """
    included=list(X.columns)
    while True:
        print(included)
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        changed=False
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
        if not changed:
            break
    return included

def top_selection(X,y,
                    liczbaZmiennych=2,
                    threshold_in=0.05):
    """ 
    Funkcja wykonuje selekcję najlepiej skorelowanych zmiennych
    na podstawie p-value z statsmodels.api.OLS
    Argumenty:
        X - pandas.DataFrame ze zmiennymi objaśniającymi
        y - Zmienna w postaci listy ze zmienną objaśnianą
        liczbaZmiennych - maksymalna liczba zmiennych która zostanie wybrana
        threshold_in - dodaj zmienną jeśli jej p-value < threshold_in
        verbose - Czy wyświetlić kolejność dodawania/usuwania zmiennych
    Zwraca: lista wyselekcjonowanych zmiennych
    Zawsze należy ustawić treshhold_in < treshold_out w celu uniknięcia zapętlenia funkcji.
    """
    included = list()
    while len(included)<liczbaZmiennych and len(included)<len(X.columns):
        changed=False
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
        if not changed:
            break
    return included

#result = stepwise_selection(X, y)

def RegresjaLiniowa(X,y):
    """ 
    Funkcja tworzy liniowy model dla wszystkich zmiennych które zostaną podane w argumencie X.
    Arguments:
        X - pandas.DataFrame ze zmiennymi objaśniającymi
        y - Zmienna objaśniana w postaci listy
    Returns: OLS.fit()
    """
    return sm.OLS(y, sm.add_constant(X)).fit()