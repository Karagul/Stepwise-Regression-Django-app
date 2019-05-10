
import matplotlib
matplotlib.use("Agg")

from django.shortcuts import redirect
import matplotlib.pyplot as plt

from io import BytesIO
import base64

from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse_lazy
from django.views import generic

import io
import pandas as pd
import numpy as np
from . import stepwise_regression as sr

def index(request):
    if request.method == 'POST':
        return redirect('dataParameters')
    return render(request, 'dataImport.html')

def dataParameters(request):
    if request.method == 'POST':
        return redirect('dataResults')
    return render(request,'dataParameters.html')

def dataResults(request):
    return render(request,'dataResults.html')
    # pokazImportDanych=True
    # pokazWyniki=False
    # wybor=''
    # metoda=''
    # dane=''
    # target=''
    # # wynik_foreward=''
    # # wynik_backward=''
    # # wynik_top=''
    # # wynik_all = ''
    # if request.method == 'POST':
    #     if request.POST['wybor'] == 'pokazWyniki': 
    #         pokazImportDanych=False
    #         pokazWyniki=True
    #         if request.FILES['plik_z_danymi']:
    #             myfile= request.FILES['plik_z_danymi']  
    #             delim=request.POST['delimiter']
    #             targetVar=int(request.POST['Zmienna_decyzyjna'])
    #             try:
    #                 plik = pd.read_csv(myfile,delimiter=delim)
    #             except :
    #                 return render(request, 'importError.html')
    #             if int(targetVar) >= len(plik.columns):
    #                 targetVar = len(plik.columns)-1
    #             X = plik.drop(plik.columns[targetVar],axis=1)
    #             dane = X.head().to_html
    #             y = plik[plik.columns[targetVar]]
    #             target = plik.columns[targetVar]

    #             alfa_in = float(request.POST['treshold_in'])
    #             alfa_out = float(request.POST['treshold_out'])

    #             alfa_for_top_selection = float(request.POST['treshold_for_top_selection'])
    #             Liczba_zmiennych = int(request.POST['Liczba_zmiennych'])
    #             request.session['dataset'] = plik.to_json()
    #             # result_foreward=sr.foreward_selection(X,y,threshold_in=alfa_in)
    #             # result_backward=sr.backward_selection(X,y,threshold_out=alfa_out)
    #             # result_top=sr.top_selection(X,y,liczbaZmiennych=Liczba_zmiennych,threshold_in=alfa_for_top_selection)
    #             # wynik_foreward=sr.RegresjaLiniowa(X[result_foreward],y).summary().as_html()
    #             # wynik_backward=sr.RegresjaLiniowa(X[result_foreward],y).summary().as_html()
    #             # wynik_top=sr.RegresjaLiniowa(X[result_top],y).summary().as_html()
    #             # wynik_all=sr.RegresjaLiniowa(X,y).summary().as_html()
    
    # return render(request,'main.html',context={
    #                                     'metoda':metoda,
    #                                     'dane': dane,
    #                                     'target':target,
    #                                     'pokazImportDanych':pokazImportDanych,
    #                                     'pokazWyniki':pokazWyniki
    #                                     }
    #                 )


# file charts.py
def simple(request):
    buf = BytesIO()
    x = np.arange(0, 5, 0.1)  
    y = np.sin(x)  
    plt.plot(x, y)
    plt.savefig(buf, format='png', dpi=300)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
    plt.close()
    buf.close()

    buf2 = BytesIO()
    x = np.arange(0, 5, 0.1)  
    y = np.cos(x)  
    plt.plot(x, y)
    plt.savefig(buf2, format='png', dpi=300)
    image_base64_2 = base64.b64encode(buf2.getvalue()).decode('utf-8').replace('\n', '')
    plt.close()
    buf2.close()
    try:
        dataset = pd.read_json(request.session['dataset']).head().to_html
    except:
        dataset = ''
    return render(request,'test.html',context={'foo':dataset,'image_base64':image_base64,'image_base64_2':image_base64_2})

class SignUp(generic.CreateView):
    form_class = UserCreationForm
    success_url = reverse_lazy('login')
    template_name = 'signup.html'