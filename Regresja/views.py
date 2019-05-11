import matplotlib
matplotlib.use("Agg")

from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse_lazy
from django.views import generic
from django.shortcuts import redirect

import io
import time
from io import BytesIO
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from . import stepwise_regression as sr

def index(request):
    if request.method == 'POST':
        
        delim=request.POST['delimiter']
        request.session['delimiter'] = delim

        myfile= request.FILES['plik_z_danymi']
        try:
            if myfile.name.endswith('.csv'):
                plik = pd.read_csv(myfile,delimiter=delim)
            elif myfile.name.endswith('.xls') or myfile.name.endswith('.xlsx'):
                plik = pd.read_excel(myfile)
            elif myfile.name.endswith('.json'):
                plik = pd.read_json(myfile)
            else:
                return render(request, 'importError.html')
        except :
            return render(request, 'importError.html')
        request.session['dataset'] = plik.to_json()

        targetVar=int(request.POST['Zmienna_decyzyjna'])
        if int(targetVar) >= len(plik.columns):
            targetVar = len(plik.columns)-1
        request.session['target_variable'] = targetVar
        return render(request,'dataParametersWaiting.html',context={'link':'dataParameters'})
    return render(request, 'dataImport.html')

def dataParameters(request):
    
    try:
        dataset = pd.read_json(request.session['dataset'])
        target_variable = request.session['target_variable']
        delimiter = request.session['delimiter']
    except:
        return redirect('index')
    if int(target_variable) >= len(dataset.columns):
        target_variable = len(dataset.columns)-1
   
    target_variable_name = dataset.columns[target_variable]
    shape_x,shape_y=dataset.shape
    dataset_y = dataset[dataset.columns[target_variable]]
    dataset_X = dataset.drop(dataset.columns[target_variable],axis=1)
    headers = dataset_X.columns
    dataset_summary = []
    if request.method == 'POST':
        if 'graph' in request.POST:
            variable = request.POST['graph']
            buf = BytesIO()
            x = dataset_X[variable]  
            y = dataset_y  
            plt.scatter(y, x)
            plt.xlabel(target_variable_name)
            plt.ylabel(variable)
            #plt.title('Wykres zmiennej '+variable)
            plt.savefig(buf, format='png', dpi=300)

            encode=base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
                
            plt.close()
            buf.close()

            context = {
                'graph':encode,
                'var':variable,
                'target_variable_name':target_variable_name
                }
            return render(request,'dataGraph.html',context=context)
        
        #############################
        #            TODO
        #############################



        return redirect('dataResults')
    try:
        for variable in headers:
            dataset_desciption = []
            dataset_desciption.append(variable)
            dataset_desciption.append(round(dataset_X[variable].count(),2))
            dataset_desciption.append(round(dataset_X[variable].mean(),2))
            dataset_desciption.append(round(dataset_X[variable].std(),2))
            dataset_desciption.append(round(dataset_X[variable].min(),2))
            dataset_desciption.append(round(dataset_X[variable].quantile(q=0.25),2))
            dataset_desciption.append(round(dataset_X[variable].quantile(q=0.5),2))
            dataset_desciption.append(round(dataset_X[variable].quantile(q=0.75),2))
            dataset_desciption.append(round(dataset_X[variable].max(),2))

            dataset_summary.append(dataset_desciption)
    except:
        return render(request,'importError.html')
    context = {
        'target_variable':target_variable_name,
        'shape_x':shape_x,
        'shape_y':shape_y,
        # 'plots':plots_images,
        'dataset_summary':dataset_summary
    }
    return render(request,'dataParameters.html',context=context)

def dataParametersGraphs(request):
    return redirect('dataParameters.html')

def dataResults(request):
    
    return render(request,'dataResults.html')

def handler404(request):
    return redirect('index')
def handler500(request):
    return redirect('index')

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
    # result_top=sr.top_selection(X,y,liczbaZmiennych=Liczba_zmiennych,threshold_in=alfa_for_top_selection)
    # wynik_foreward=sr.RegresjaLiniowa(X[result_foreward],y).summary().as_html()
    # wynik_backward=sr.RegresjaLiniowa(X[result_foreward],y).summary().as_html()
    # wynik_top=sr.RegresjaLiniowa(X[result_top],y).summary().as_html()
    # wynik_all=sr.RegresjaLiniowa(X,y).summary().as_html()
    
    # return render(request,'main.html',context={
    #                                     'metoda':metoda,
    #                                     'dane': dane,
    #                                     'target':target,
    #                                     'pokazImportDanych':pokazImportDanych,
    #                                     'pokazWyniki':pokazWyniki
    #                                     }
    #                 )

class SignUp(generic.CreateView):
    form_class = UserCreationForm
    success_url = reverse_lazy('login')
    template_name = 'signup.html'