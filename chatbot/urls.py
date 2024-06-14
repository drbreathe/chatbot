from django.urls import path
from . import views


urlpatterns = [
    #path(website link, linking to view, name )
    path('', views.login, name='login'),
    #path('chatbot', views.bot_prompt, name='chatbot'),
    path('chatbot', views.bot_rag, name='chatbot'),
    path('register', views.register, name='register'),
    path('logout', views.logout, name='logout'),
    path('clear_session', views.clear_session, name='clear_session'),
    
    # If you change any api urls, please update the documentation in Postman accordingly
    #path('api/login/', views.api_login, name='api_login'),SECRET_KEY
    path('api/register/', views.api_register, name='api_register'),
    #path('api/chat/', views.api_bot_prompt, name='api_register'),
    path('api/logout/', views.api_logout, name='api_logout'),
    path('api/clear_session/', views.api_clear_session, name='api_clear_session'),
]