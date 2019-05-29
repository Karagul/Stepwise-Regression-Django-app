from django.shortcuts import redirect
from django.views import generic
from django.urls import reverse_lazy,reverse
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.utils.translation import gettext, gettext_lazy as _
from django.shortcuts import render

import pandas as pd
from . import stepwise_regression as sr
from statsmodels.api import add_constant


def index(request):
    # POST jeśli plik został przesłany
    if request.method == 'POST':
        # zapis delimitera w sesji
        request.session['delimiter'] = request.POST['delimiter']
        # zapisanie przesłanego pliku w sesji oraz sprawdzenie poprawności
        myfile = request.FILES['plik_z_danymi']
        # próba odczytania danych z pliku lub zwrócenie informacji o błędzie
        try:
            if myfile.name.endswith(('.csv', '.txt')):
                myfile_dataFrame = pd.read_csv(myfile, delimiter=request.POST['delimiter'])
            else:
                return render(request, 'importError.html')
        except:
            # zwrócenie komunikatu o błędzie w przypadku błędu przy imporcie z pliku
            return render(request, 'importError.html')

        request.session['dataset'] = myfile_dataFrame
        # zapis w sesji która zmienna jest zależna, oraz sprawdzenie czy liczba nie jest zbyt duża
        targetVar = int(request.POST['Zmienna_decyzyjna'])
        if targetVar >= len(myfile_dataFrame.columns):
            targetVar = len(myfile_dataFrame.columns)-1
        request.session['target_variable'] = targetVar
        # po zapisaniu danych w sesji przejście do kolejnego widoku i analizy zmiennych
        return render(request, 'dataParametersWaiting.html', context={'link': reverse('dataParameters')})

    # wyświetlenie widoku importu, jesli myfile_dataFrame nie został jeszcze przesłany
    return render(request, 'dataImport.html')

# widok wywoływany po przejściu do /dataParameters
def dataParameters(request):

    # sprawdzenie czy plik został przesłany
    if 'dataset' not in request.session:
        return redirect('index')

    # przypisanie zmiennych dotyczących zbioru danych
    dataset = request.session['dataset']
    target_variable = request.session['target_variable']
    delimiter = request.session['delimiter']

    target_variable_name = dataset.columns[target_variable]
    shape_x, shape_y = dataset.shape
    dataset_y = dataset[dataset.columns[target_variable]]
    dataset_X = dataset.drop(dataset.columns[target_variable], axis=1)
    headers = dataset_X.columns

    # po przesłaniu odpowiedzi
    if request.method == 'POST':

        # po wciśnięciu przyciku 'WYKRES'
        if 'graph' in request.POST:
            variable = request.POST['graph']

            # przygotowanie  wykresu punktowego
            plot = sr.prepare_scatter_plot(
                dataset_X[variable], dataset_y, variable, target_variable_name)
            # i wykresu pudełkowego
            boxplot = sr.prepare_box_plot(dataset_X[variable])
            context = {
                'graph': plot,
                'boxplot': boxplot,
                'var': variable,
                'target_variable_name': target_variable_name
            }

            return render(request, 'dataGraph.html', context=context)

        # po wciśnięciu przycisku 'PRZEJDŹ DO WYNIKÓW'
        else:
            # zapisanie parametrów w sesji i przejście do widoku z wynikami
            request.session['treshold_in'] = float(request.POST['treshold_in'])
            request.session['treshold_out'] = float(
                request.POST['treshold_out'])
            request.session['number_of_variables'] = int(
                request.POST['Liczba_zmiennych'])

            return redirect('dataResults')

    # stworzenie macierzy korelacji i zapisanie jej w html
    corr_matrix = dataset.corr().round(3).to_html(classes='table',border='0', justify='center')
    # corr_matrix = sr.correlationPlot(corr_matrix)
    try:
        # obliczenie statystyk opisowych z pomocą list comprahension
        dataset_summary = sr.dataset_statistic_summary(dataset_X, headers)
        # błąd gdy w pliku znajdują się zmienne nieliczbowe
    except:
        return render(request, 'importError.html')

    context = {
        'target_variable': target_variable_name,
        'shape_x': shape_x,
        'shape_y': shape_y,
        'corr_matrix': corr_matrix,
        'dataset_summary': dataset_summary
    }
    return render(request, 'dataParameters.html', context=context)

# widok wywoływany po przekierowaniu do /dataResults


def dataResults(request):

    # sprawdzenie czy dataset został załadowany w sesji
    if 'dataset' not in request.session:
        return redirect('index')

    # przypisanie zmiennych z sesji
    dataset = request.session['dataset']
    target_variable = request.session['target_variable']
    treshold_in = request.session['treshold_in']
    treshold_out = request.session['treshold_out']
    number_of_variables = request.session['number_of_variables']

    # podział datasetu na zbiór zmiennych objaśniających X i zmienną objaśnianą
    dataset_y = dataset[dataset.columns[target_variable]]
    dataset_X = dataset.drop(dataset.columns[target_variable], axis=1)

    # podział danych na zbiory train i test - test size i random state są ustalone przy implementacji metody
    X_train, X_test, y_train, y_test = sr.train_test_split_with_params(
        dataset_X, dataset_y)

    # ustalenie zmiennych wchodzących do modeli czterema metodami
    result_forward = sr.forward_selection(
        X_train, y_train, threshold_in=treshold_in)
    result_backward = sr.backward_selection(
        X_train, y_train, threshold_out=treshold_out)
    result_top = list(sr.top_selection(
        X_train, y_train, var_number=number_of_variables))
    result_all = list(X_test.columns)

    # przygotowanie modeli ze zmiennymi ustalonymi w poprzednim kroku
    ols_forward = sr.linear_regression(X_train[result_forward], y_train)
    ols_backward = sr.linear_regression(X_train[result_backward], y_train)
    ols_top = sr.linear_regression(X_train[result_top], y_train)
    ols_all = sr.linear_regression(X_train, y_train)

    # oszacowanie wartości y na zbiorze testowym oraz dodanie wyrazu wolnego
    y_predict_forward = ols_forward.predict(
        add_constant(X_test[result_forward]))
    y_predict_backward = ols_backward.predict(
        add_constant(X_test[result_backward]))
    y_predict_top = ols_top.predict(add_constant(X_test[result_top]))
    y_predict_all = ols_all.predict(add_constant(X_test))

    # przygotowanie podsumowania dla wszystkich metod
    forward_summary = sr.ols_sum_table(
        y_test, y_predict_forward, ols_forward, result_forward, 'Forward Selection')
    backward_summary = sr.ols_sum_table(
        y_test, y_predict_backward, ols_backward, result_backward, 'Backward Selection')
    top_summary = sr.ols_sum_table(
        y_test, y_predict_top, ols_top, result_top, 'Najlepiej skorelowane zmienne')
    all_summary = sr.ols_sum_table(
        y_test, y_predict_all, ols_all, result_all, 'Wszystkie zmienne objaśniające')

    # zapisanie wszystkich podsumowań w jednej tablicy
    summary_of_all_methods = [forward_summary,
                              backward_summary, 
                              top_summary, 
                              all_summary]

    return render(request, 'dataResult.html', context={
        'summary_all': summary_of_all_methods,
        'number_of_variables': number_of_variables,
    })

# klasa obsługująca rejestrację użytkownika
# class UserCreationForm
class SignUp(generic.CreateView):
    form_class = UserCreationForm
    success_url = reverse_lazy('login')
    template_name = 'signup.html'

# obsługa błędów 404 oraz 500 - przekierowanie na stronę główną

def handler404(request):
    return redirect('index')


def handler500(request):
    return redirect('index')