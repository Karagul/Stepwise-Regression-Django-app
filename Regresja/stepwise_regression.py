import matplotlib
matplotlib.use("Agg")

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import base64
import io
import statsmodels.api as sm
import numpy as np
import pandas as pd



# USTAWIENIE WSTĘPNYCH PARAMETRÓW
# None - losowo, Int - określony seed, powtarzalne wyniki
SEED_VALUE = 1
# Należy również ustawić do ilu miejsc po przecinku zwracać wyniki
APPROX = 3
# Stosunek podziału train/test, wartość - % zbioru testowego
TEST_VALUE = 0.2


def forward_selection(X, y, threshold_in=0.05):
    """ 
    Funkcja wykonuje selekcję forward zmiennych 
    na podstawie p-value z statsmodels.api.OLS
    Argumenty:
        X - tablica ze zmiennymi objaśniającymi (pandas.Dataframe)
        y - Zmienna w postaci listy ze zmienną objaśnianą (array)
        threshold_in - dodaj zmienną jeśli jej p-value < threshold_in (decimal)
        verbose - Czy wyświetlić kolejność dodawania/usuwania zmiennych (boolean)
    Zwraca: lista wyselekcjonowanych zmiennych
    """
    # zaczynamy z pustą listą
    included = list()
    while True:
        changed = False
        # lista zmiennych które nie zostały wprowadzone do modelu
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(
                pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
        if not changed:
            break
    return included


def backward_selection(X, y, threshold_out=0.05):
    """ 
    Funkcja wykonuje selekcję backward zmiennych 
    na podstawie p-value z statsmodels.api.OLS
    Argumenty:
        X - tablica ze zmiennymi objaśniającymi (pandas.Dataframe)
        y - Zmienna w postaci listy ze zmienną objaśnianą (array)
        threshold_out - algorytm usuwa zmienną jeśli jej p-value > threshold_in (decimal)
        verbose - Czy wyświetlić kolejność dodawania/usuwania zmiennych (boolean)
    Zwraca: lista wyselekcjonowanych zmiennych
    """
    # zaczynamy z listą wszystkich zmiennych
    included = list(X.columns)
    while True:
        # przygotowanie modelu ze zmiennymi included
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        changed = False
        # wybranie wartośći pvalue - poza 1
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()

        # usunięcie zmiennej jeśli jest powyżej treshholdu
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)

        # koniec pętli jeśli nie było żadnej zmiennej z treshholdem powyżej treshholdu
        if not changed:
            break
    return included


def top_selection(X, y, var_number=2):
    """ 
    Funkcja wykonuje selekcję najlepiej skorelowanych zmiennych
    Argumenty:
        X - pandas.DataFrame ze zmiennymi objaśniającymi
        y - Zmienna w postaci listy ze zmienną objaśnianą
        liczbaZmiennych - Liczba zmiennych która zostanie wybrana
    Zwraca: lista wyselekcjonowanych zmiennych
    """
    # umieszczenie zmiennej objaśnianej na końcu dataframe
    dataset = X.assign(DependendVariable=y)
    # obliczenie macierzy korelacji i pobranie modułów z ostatniej kolumny - czyli korelacji ze zmienną y
    corr_array = dataset.corr().iloc[:-1, -1].abs()
    # sortowanie i wybranie najlepiej skorelowanych zmiennych
    results = corr_array.sort_values(ascending=False)[0:var_number]
    return results.index


def linear_regression(X, y):
    """ 
    Funkcja tworzy liniowy model dla wszystkich zmiennych które zostaną podane w argumencie X.
    Fukcja dodaje wyraz wolny i dopasowywuje model.
    Arguments:
        X - pandas.DataFrame ze zmiennymi objaśniającymi
        y - Zmienna objaśniana w postaci listy
    Returns: OLS.fit()
    """
    return sm.OLS(y, sm.add_constant(X)).fit()


def dataset_statistic_summary(dataset, headers):
    """ 
    Funkcja oblicza szereg statystyk opisowych dla przesłanego dataframe.
    Argumenty:
        X - pandas.DataFrame ze zmiennymi objaśniającymi
        headers - nazwy zmiennych w X
    Zwraca: [[nazwa zmiennej,liczebność,średnia, odchylenie, min, kwartyl 0.25, mediana, kwartyl 0.75, max]]
    """
    return [
        [variable,
         round(dataset[variable].count(), APPROX),
         round(dataset[variable].mean(), APPROX),
         round(dataset[variable].std(), APPROX),
         round(dataset[variable].min(), APPROX),
         round(dataset[variable].quantile(q=0.25), APPROX),
         round(dataset[variable].quantile(q=0.5), APPROX),
         round(dataset[variable].quantile(q=0.75), APPROX),
         round(dataset[variable].max(), APPROX)] for variable in headers]


def ols_sum_table(y_true, y_pred, ols_model, headers, method_name):
    '''
    Funkcja przygotowywuje tablicę ze statystykami oszacowanego modelo
    Argumenty:
        y_true - rzeczywiste obserwacje zmiennej
        y_pred - prognozowane obserwacje zmiennej
        ols_model - model przygotowany za pomoca statmodels.api.OLS.fit()
        method_name - nazwa metody doboru zmiennych do modelu
        headers - nazwy zmiennych decyzyjnych
    Zwraca: wartość MAPE 
    '''
    # dodanie do nagłówków oznaczenia wyrazu wolnego
    headers.insert(0, 'CONST')
    # zwrocenie listy z poszczególnymi statystykami
    return [
        list(ols_model.params.values.round(APPROX)),    # oszacowania parametrów
        list(ols_model.pvalues.values.round(APPROX)),   # lista z wartośćiami p-value
        ols_model.rsquared.round(APPROX),               # wartość r^2
        ols_model.rsquared_adj.round(APPROX),           # wartość skorygowanego r^2
        mean_error(y_true,y_pred),                      # ME - błąd średni
        root_mean_squared_error(y_true,y_pred),         # RMSE - pierwiastek błędu średniokwadratowego
        mean_absolute_error(y_true, y_pred),            # MAE - średni błąd absolutny
        mean_absolute_percentage_error(y_true, y_pred), # MAPE - średni procentowy błąd absolutny
        headers,                                        # nazwy zmiennych
        method_name                                     # nazwa metody
    ]

def mean_error(y_true, y_pred, approx=2):
    '''
    ME - Błąd średni
    '''
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    result = np.mean(y_true - y_pred)
    return result.round(approx)

def root_mean_squared_error(y_true, y_pred, approx=2):
    '''
    RMSE - błąd średniokwadratowy
    '''
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    result = np.sqrt(np.mean((y_true - y_pred)**2))
    return result.round(approx)

def mean_percentage_error(y_true, y_pred):
    '''
    MPE - procentowy błąd średni
    '''
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    result = np.mean((y_true - y_pred)/y_true)*1000
    return result.round(approx)
def mean_absolute_error(y_true, y_pred):
    '''
    Funkcja oblicza średni absolutny błąd prognozy.
    Argumenty:
        y_true - rzeczywiste obserwacje zmiennej
        y_pred - prognozowane obserwacje zmiennej
    Zwraca: wartość MAE 
    '''
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    result = np.mean(np.abs(y_true - y_pred))
    return result.round(APPROX)


def mean_absolute_percentage_error(y_true, y_pred):
    '''
    Funkcja oblicza średni procentowy absolutny błąd prognozy.
    Argumenty:
        y_true - rzeczywiste obserwacje zmiennej
        y_pred - prognozowane obserwacje zmiennej
    Zwraca: wartość MAPE 
    '''
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    result = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return result.round(APPROX)


def convert_plot_to_bse64():
    '''
    Funkcja do działania wymaga wcześniejszego stworzenia grafu za pomocą matplotlib.
    Po przekonwertowaniu wykresu, narysowany wykres zostaje wyczyszczony.
    '''
    # otworzenie bufora
    buf = io.BytesIO()
    # zapisanie figury do bufora
    plt.savefig(buf, format='png', dpi=300)
    # zakodowanie danych z bufora do base64 w celu łatwiejszego eksportu do html
    encode = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
    # zamknięcie figury oraz bufora
    buf.close()
    plt.close()
    return encode


def prepare_scatter_plot(x, y, x_name, y_name):

    plt.style.use('seaborn')
    plt.scatter(x, y)
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    # konwersja do formatu base64 i zapisanie wykresu zmiennej
    result = convert_plot_to_bse64()
    return result


def correlationPlot(corr_matrix):

    corr = corr_matrix
    plt.figure(figsize=(len(corr.columns), len(corr.columns)/2))
    plt.matshow(corr_matrix)

    # konwersja do formatu base64 i zapisanie wykresu zmiennej
    correlation_matrix = convert_plot_to_bse64()

    return correlation_matrix


def prepare_box_plot(dataset):
    plt.boxplot(dataset)
    # konwersja do formatu base64 i zapisanie wykresu zmiennej
    boxplot = convert_plot_to_bse64()
    return boxplot


def train_test_split_with_params(dataset_X, dataset_y):
    return train_test_split(
        dataset_X, dataset_y, 
        test_size=TEST_VALUE, 
        random_state=SEED_VALUE)
