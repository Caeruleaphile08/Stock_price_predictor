import requests
from django.shortcuts import render, redirect
from .models import CustomUser
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import get_user_model
from django.contrib import messages
from django.contrib.auth.tokens import default_token_generator
import matplotlib
matplotlib.use('Agg')  
import os
import matplotlib.pyplot as plt  # Add this line
from django.shortcuts import render
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .utils import fetch_stock_data, preprocess_data, train_model, predict_stock_price, visualize_results_and_save
from keras.models import load_model



CustomUser = get_user_model()

def land_page(request):
    return render(request, 'land_page.html')

def market(request):
    return render(request, 'market.html')

def error(request):
    return render(request, 'error.html')

@csrf_exempt
def signup(request):
    if request.method == 'POST':
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        email = request.POST['email']
        password = request.POST['password']
        confirm_password = request.POST['confirm_password']

        if password == confirm_password:
            if CustomUser.objects.filter(email=email).exists():
                messages.error(request, 'Email already exists. Please use a different email.')
                return redirect('Home:signup') 
            else:
                user = CustomUser.objects.create_user(email=email, password=password, first_name=first_name, last_name=last_name)
                login(request, user)
                return redirect('Home:login')  
        else:
            messages.error(request, 'Passwords do not match.')
    return render(request, 'signup.html')

def login_view(request):
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']

        user = authenticate(request, email=email, password=password)

        if user is not None:
            login(request, user)
            return redirect('Home:market')  
        else:
            messages.error(request, "Invalid credentials")
            return redirect('Home:login')
    else:
        return render(request, 'login.html')
@csrf_exempt
@login_required
def profile(request):
    if request.method == 'POST':
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('email')

        if email and CustomUser.objects.exclude(id=request.user.id).filter(email=email).exists():
            messages.error(request, 'Email already exists. Please use a different email.')
            return redirect('Home:profile')

        request.user.first_name = first_name
        request.user.last_name = last_name
        request.user.email = email
        request.user.save()

        messages.success(request, 'Profile updated successfully!')
        return redirect('Home:profile')

    return render(request, 'profile.html', {'user': request.user})  # Return the profile template with the current user data

def live_graph(request):
    symbol = request.GET.get('symbol', 'TATAMOTORS')  # Default symbol is 'TATAMOTORS'
    
    # Fetch stock details from Alpha Vantage API
    api_key = '3L4IET2PBT2JJYZ7'
    api_url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}'
    
    response = requests.get(api_url)
    data = response.json()
    
    stock_data = {
        #'volume': data['Global Quote']['06. volume'],
        #'adj_close': data['Global Quote']['05. price'],
        #'today_low': data['Global Quote']['04. low'],
        #'today_high': data['Global Quote']['03. high'],
        #'52_week_low': data['Global Quote']['52_week_low'],
        #'52_week_high': data['Global Quote']['52_week_high'],
    }
    
    context = {
        'symbol': symbol,
        'stock_data': stock_data,
    }
    
    return render(request, 'live_graph.html', context)


def predict_graph(request):
    if request.method == 'POST':
        symbol = request.POST.get('symbol')
       
        # Fetch stock data
        stock_data = fetch_stock_data(symbol)
       
        # Preprocess data
        features, target = preprocess_data(stock_data)
       
        # Normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)
       
        # Reshape data for LSTM model
        X = np.reshape(scaled_features, (scaled_features.shape[0], 1, scaled_features.shape[1]))
       
        # Load model
        model, _, _, _ = train_model(features, target) 
        #model, scaler = train_model(features, target) 
        # Predict stock price
        predicted_prices = predict_stock_price(model, X)       
        # Visualize the results and save the graph as an image
        graph_image_path = visualize_results_and_save(stock_data.index, target, predicted_prices)
       
        context = {
            'predicted_prices':predicted_prices[0],
            'symbol': symbol,
            'graph_image_path': os.path.join('/static/', graph_image_path),
        }
        return render(request, 'predict_graph.html', context)
    else:
        return render(request, 'predict_graph.html')

