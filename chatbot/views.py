import sys
import os
import re
sys.path.append('../..')
# Get the directory where the script is located
current_directory = os.path.abspath(os.path.dirname(__file__))

# Add this directory to the system path
if current_directory not in sys.path:
    sys.path.append(current_directory)
from django import forms
from django.contrib.auth import authenticate, login as auth_login
from django.contrib.auth import logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.contrib import auth
from django.contrib.auth.models import User
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from django.shortcuts import render, redirect
from django.http import JsonResponse
import openai
from openai import OpenAI
from chatbot.rag_bot import RagBot
from .models import Chat
from django.utils import timezone
from django.contrib import messages
from tqdm import tqdm
from django.core.cache import cache
from langchain.memory import ConversationSummaryMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory

# REST API VERSION Version

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate
from .serializers import ChatSerializer
from django.core.exceptions import ObjectDoesNotExist
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rag_bot import get_chatbot_memory

openai.api_key = os.environ['OPENAI_API_KEY']

# WEB APP VERSION

# Create your views here.
client = OpenAI()

# Define the SignupForm
class SignupForm(forms.Form):
    username = forms.CharField(min_length=6, max_length=100)
    email = forms.EmailField()
    password1 = forms.CharField(widget=forms.PasswordInput, min_length=8)
    password2 = forms.CharField(widget=forms.PasswordInput, min_length=8)
    dob = forms.DateField(required=True)
    gender = forms.ChoiceField(choices=[('male', 'Male'), ('female', 'Female'), ('other', 'Other')], required=True)

    def clean_email(self):
        email = self.cleaned_data['email']
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError('Email address already in use.')
        return email

    def clean_username(self):
        username = self.cleaned_data['username']
        if User.objects.filter(username=username).exists():
            raise forms.ValidationError('Username already exists.')
        return username

    def clean(self):
        cleaned_data = super().clean()
        password1 = cleaned_data.get('password1')
        password2 = cleaned_data.get('password2')
        
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError("Passwords don't match.")
        return cleaned_data


def load_chat_history(memory):
    if isinstance(memory, ConversationSummaryMemory):
        return memory.buffer
    if isinstance(memory, (ConversationBufferWindowMemory)):
        return memory.buffer_as_str
    if isinstance(memory, (ConversationSummaryBufferMemory)):
        memory.prune()
        if memory.moving_summary_buffer == '':
            test = {}
            chat_history = memory.load_memory_variables(test)
            if 'history' in chat_history:
                return chat_history['history']
            else:
                return memory.moving_summary_buffer
        else:
            return memory.moving_summary_buffer


def get_memory():
    memory = cache.get('chatbot_memory')
    if not memory:
        memory = get_chatbot_memory(buffer_window=4, smry_bfr_tkn=30, memory_type='buffer')
        cache.set('chatbot_memory', memory, timeout=None)  # Adjust timeout as necessary
    return memory

@login_required
@require_http_methods(["GET", "POST"])
def bot_rag(request):
    # Limit the number of chats loaded for performance considerations
    memory = get_memory()
    
    rag_bot_instance  = RagBot(prompt_hub_qa = "drbreaths/rag_prompt2",
                               prompt_hub_condense= "drbreaths/drbreaths_condense_question_prompt2")
    
    chats = Chat.objects.filter(user=request.user, visible=1).order_by('-created_at')[:50]

    # Handling for POST requests - Processing chat messages
    if request.method == 'POST':
        user_message = request.POST.get('message', '').strip()
        if user_message:
            # Attempt to load memory from database for a new state
            if not memory.load_memory_variables({})["chat_history"]:
                for chat in chats:
                    if chat.message and chat.response:
                        memory.save_context({"input": chat.message}, {"output": chat.response})
            
            ai_response = rag_bot_instance.chat({"messages": user_message} , history= load_chat_history(memory))
            memory.save_context({"input": user_message}, {"output": ai_response})

            if ai_response is not None:
                chat = Chat(user=request.user, message=user_message, response=ai_response, created_at=timezone.now())
                chat.save()
                return JsonResponse({'message': user_message, 'response': ai_response})
            else:
                # Handle case where response is None
                return JsonResponse({'message': user_message, 'error': 'Bot failed to generate a response.'})
        else:
            # Handle case where user_message is empty or only whitespace
            return JsonResponse({'error': 'No message provided.'})

    # Handling for GET requests - Initially loading the chat interface
    return render(request, 'chatbot.html', {'chats': chats})

def login(request):
    if request.method == 'POST':
        username, password = request.POST['username'], request.POST['password']
        user = auth.authenticate(request, username= username, password = password)
        if user is not None:
            auth.login(request, user)
            return redirect('chatbot')
        else:
            error_message = 'Invalid user name or password'
            return render(request, 'login.html', {'error_message': error_message})
    else:
        return render(request, 'login.html')

def register(request):
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            email = form.cleaned_data.get('email')
            password1 = form.cleaned_data.get('password1')
            # No need to check if the passwords match, the form validation does this
            
            # Create the user
            user = User.objects.create_user(username=username, email=email, password=password1)
            user.save()

            # Authenticate and login the user
            user = authenticate(request, username=username, password=password1)
            auth_login(request, user)
            messages.success(request, "Registration successful, you are now logged in.")
            return redirect('chatbot')  # Redirect to the desired page after successful registration
        else:
            # Pass the form instance back to the template if the form is invalid
            print(form.errors)
            return render(request, 'register.html', {'form': form})
    else:
        form = SignupForm(initial={'username': '', 'email': ''})

    return render(request, 'register.html', {'form': form})

def logout(request):
    auth_logout(request)
    return redirect('login')

def clear_session(request):
    # Set the visibility flag to False for the user's chat messages
    Chat.objects.filter(user=request.user).update(visible=False)
    # prompt_bot1.memory.clear()
    # rag_bot1.memory.clear()
    return redirect('chatbot')

@api_view(['POST'])
def api_login(request):
    username = request.data.get('username')
    password = request.data.get('password')
    user = authenticate(username=username, password=password)
    if user:
        token, _ = Token.objects.get_or_create(user=user)
        return Response({'token': token.key})
    else:
        return Response({'error': 'Invalid username or password'}, status=400)
    
@api_view(['POST'])
def api_register(request):
    username = request.data.get('username')
    email = request.data.get('email')
    password1 = request.data.get('password1')
    password2 = request.data.get('password2')

    # Check for empty fields
    if not username:
        return Response({'error': 'Username is required.'}, status=400)
    if not email:
        return Response({'error': 'Email is required.'}, status=400)
    if not password1:
        return Response({'error': 'Password is required.'}, status=400)
    if not password2:
        return Response({'error': 'Confirm password is required.'}, status=400)
    
    # Email validation for proper format
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email) or not email.endswith('@gmail.com'):
        return Response({'error': 'Invalid or non-Gmail address.'}, status=400)

    # Check for uniqueness of username and email
    if User.objects.filter(username=username).exists():
        return Response({'error': 'Username already exists'}, status=400)
    if User.objects.filter(email=email).exists():
        return Response({'error': 'Email address already in use'}, status=400)

    # Password validation for match and length
    if password1 != password2:
        return Response({'error': "Passwords don't match"}, status=400)
    if len(password1) < 8:
        return Response({'error': 'Password must be at least 8 characters long.'}, status=400)

    # Attempt to create the user
    try:
        user = User.objects.create_user(username, email, password1)
        token, _ = Token.objects.get_or_create(user=user)
        return Response({'token': token.key}, status=201)  # Return HTTP 201 for created resource
    except Exception as e:
        # Log the error or handle it as appropriate
        return Response({'error': f'Error creating account: {str(e)}'}, status=400)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def api_logout(request):
    try:
        request.user.auth_token.delete()
    except (AttributeError, ObjectDoesNotExist):
        return Response({'error': 'Invalid token.'}, status=status.HTTP_400_BAD_REQUEST)
    
    return Response({'success': 'Successfully logged out.'})

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def api_clear_session(request):
    Chat.objects.filter(user=request.user).update(visible=False)
    #prompt_bot1.memory.clear()
    return Response({'success': 'Chat history cleared.'})