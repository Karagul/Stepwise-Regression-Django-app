import matplotlib
matplotlib.use("Agg")
from django.shortcuts import render
from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse_lazy
from django.views import generic
from django.shortcuts import redirect
import io
import sys
import time
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from . import stepwise_regression as sr
import statsmodels.api as sm


# Czy ustawić random state?
# None - losowo, Int - określony seed, powtarzalne wyniki
seed_value=1
# Należy również ustawić do ilu miejsc po przecinku zwracać wyniki
approximation=3
# Stosunek podziału train/test, wartość - % zbioru testowego
test_value=0.2


def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    result = np.mean(np.abs(y_true - y_pred))*100
    return result.round(3)
    
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    result = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return result.round(3)

def ols_sum_table(y_true,y_pred,ols_model,method_name,headers):
    result=[]
    result.append(method_name)
    result.append(list(ols_model.params.values.round(approximation)))
    result.append(ols_model.rsquared.round(approximation))
    result.append(ols_model.rsquared_adj.round(approximation))
    result.append(list(ols_model.pvalues.values.round(approximation)))
    result.append(list(ols_model.bse.values.round(approximation)))
    result.append(mean_absolute_error(y_true,y_pred))
    result.append(mean_absolute_percentage_error(y_true,y_pred))
    headers.insert(0,'CONST')
    result.append(headers)
    return result




def convertDataToPlot(x,y,x_name,y_name):
    buf = io.BytesIO()
    plt.style.use('seaborn')
    plt.scatter(x, y)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    #plt.title('Wykres zmiennej '+variable)
    plt.savefig(buf, format='png', dpi=300)
    encode = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
    plt.close()
    buf.close()
    return encode

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
            encode = convertDataToPlot(dataset_X[variable],dataset_y,variable,target_variable_name)

            context = {
                'graph':encode,
                'var':variable,
                'target_variable_name':target_variable_name
                }
            return render(request,'dataGraph.html',context=context)
        

        request.session['treshold_in'] = float(request.POST['treshold_in'])
        request.session['treshold_out'] = float(request.POST['treshold_out'])
        request.session['number_of_variables'] = int(request.POST['Liczba_zmiennych'])

        return redirect('dataResults')
    try:
        for variable in headers:
            dataset_desciption = []
            dataset_desciption.append(variable)
            dataset_desciption.append(round(dataset_X[variable].count(),approximation))
            dataset_desciption.append(round(dataset_X[variable].mean(),approximation))
            dataset_desciption.append(round(dataset_X[variable].std(),approximation))
            dataset_desciption.append(round(dataset_X[variable].min(),approximation))
            dataset_desciption.append(round(dataset_X[variable].quantile(q=0.25),approximation))
            dataset_desciption.append(round(dataset_X[variable].quantile(q=0.5),approximation))
            dataset_desciption.append(round(dataset_X[variable].quantile(q=0.75),approximation))
            dataset_desciption.append(round(dataset_X[variable].max(),approximation))
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
    try:
        dataset = pd.read_json(request.session['dataset'])
        target_variable = request.session['target_variable']
        treshold_in = request.session['treshold_in'] 
        treshold_out = request.session['treshold_out']
        number_of_variables = request.session['number_of_variables']
    except:
        return redirect('index')
    
    dataset_y = dataset[dataset.columns[target_variable]]
    dataset_X = dataset.drop(dataset.columns[target_variable],axis=1)

    X_train, X_test, y_train, y_test = train_test_split(dataset_X,dataset_y, test_size=test_value,random_state=seed_value)

    result_forward=sr.forward_selection(X_train,y_train,threshold_in=treshold_in)
    ols_forward = sr.linear_regression_sm(X_train[result_forward],y_train)
    y_predict_forward = ols_forward.predict(sm.add_constant(X_test[result_forward]))
    forward_summary = ols_sum_table(y_test,y_predict_forward,ols_forward,'Forward Selection',result_forward)


    result_backward=sr.backward_selection(X_train,y_train,threshold_out=treshold_out)
    ols_backward = sr.linear_regression_sm(X_train[result_backward],y_train)
    y_predict_backward = ols_backward.predict(sm.add_constant(X_test[result_backward]))
    backward_summary = ols_sum_table(y_test,y_predict_backward,ols_backward,'Backward Selection', result_backward)
    
    result_top=list(sr.top_selection(X_train,y_train,var_number=number_of_variables))
    ols_top = sr.linear_regression_sm(X_train[result_top],y_train)
    y_predict_top = ols_top.predict(sm.add_constant(X_test[result_top]))
    top_summary = ols_sum_table(y_test,y_predict_top,ols_top,str(number_of_variables)+' skorelowanych zmiennych',result_top)
    
    
    result_all = list(X_test.columns)
    ols_all = sr.linear_regression_sm(X_train,y_train)
    y_predict_all = ols_all.predict(sm.add_constant(X_test))
    all_summary = ols_sum_table(y_test,y_predict_all,ols_all,'wszystkie zmienne',result_all)

    summary_of_all_methods = [forward_summary,backward_summary,top_summary,all_summary]


    return render(request,'dataResult.html',context={
        'summary_all':summary_of_all_methods,
        'number_of_variables':number_of_variables,

    })

class SignUp(generic.CreateView):
    form_class = UserCreationForm
    success_url = reverse_lazy('login')
    template_name = 'signup.html'

def handler404(request):
    return redirect('index')

def handler500(request):
    return redirect('index')


