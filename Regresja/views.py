
from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse_lazy
from django.views import generic

import io
import pandas as pd
from . import stepwise_regression as sr



def index(request):

    pokazImportDanych=True
    pokazWyniki=False
    wybor=''
    metoda=''
    dane=''
    target=''
    wynik_stepwise=''
    wynik_top=''
    wynik_all = ''

    if request.method == 'POST':

        if request.POST['wybor'] == 'pokazWyniki': 
            pokazImportDanych=False
            pokazWyniki=True
            if request.FILES['plik_z_danymi']:
                try:
                    myfile= request.FILES['plik_z_danymi']
                    delim=request.POST['delimiter']
                    targetVar=int(request.POST['Zmienna_decyzyjna'])
                    plik = pd.read_csv(myfile,delimiter=delim)
                    if targetVar >= len(plik.columns):
                        targetVar = len(plik.columns)-1
                    X = plik.drop(plik.columns[targetVar],axis=1)
                    dane = X.head().to_html
                    y = plik[plik.columns[targetVar]]
                    target = plik.columns[targetVar]

                    alfa_in = float(request.POST['treshold_in'])
                    alfa_out = float(request.POST['treshold_out'])

                    alfa_for_top_selection = float(request.POST['treshold_for_top_selection'])
                    Liczba_zmiennych = int(request.POST['Liczba_zmiennych'])

                    result_stepwise=sr.stepwise_selection(X,y,threshold_in=alfa_in,threshold_out=alfa_out)
                    result_top=sr.top_selection(X,y,liczbaZmiennych=Liczba_zmiennych,threshold_in=alfa_for_top_selection)
                    wynik_stepwise=sr.RegresjaLiniowa(X[result_stepwise],y).as_html()
                    wynik_top=sr.RegresjaLiniowa(X[result_top],y).as_html()
                    wynik_all=sr.RegresjaLiniowa(X,y).as_html()
                except :
                    return HttpResponse('Coś poszło nie tak, prawdopodobnie plik zawiera błędy :( <br> <a href="http://127.0.0.1:8000/">Powrót</a>')

    return render(request,'main.html',context={
                                        'metoda':metoda,
                                        'dane': dane,
                                        'target':target,
                                        'pokazImportDanych':pokazImportDanych,
                                        'pokazWyniki':pokazWyniki,
                                        'wynik_stepwise':wynik_stepwise,
                                        'wynik_top':wynik_top,
                                        'wynik_all':wynik_all,
                                        })


class SignUp(generic.CreateView):
    form_class = UserCreationForm
    success_url = reverse_lazy('login')
    template_name = 'signup.html'