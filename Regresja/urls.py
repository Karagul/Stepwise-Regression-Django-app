from django.urls import path, include
from . import views

urlpatterns = [

    path('',views.index, name='index'),
    path('', include('django.contrib.auth.urls')),
    path('signup/', views.SignUp.as_view(), name='signup'),
    path('dataParameters/',views.dataParameters, name='dataParameters'),
    path('dataResults/',views.dataResults, name='dataResults')
]

# login/ [name='login']
# logout/ [name='logout']
# password_change/ [name='password_change']
# password_change/done/ [name='password_change_done']
# password_reset/ [name='password_reset']
# password_reset/done/ [name='password_reset_done']
# reset/<uidb64>/<token>/ [name='password_reset_confirm']
# reset/done/ [name='password_reset_complete']