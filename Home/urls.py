from django.conf import settings
from django.urls import path
from . import views
from django.contrib.auth import views as auth_views
from django.conf.urls.static import static

app_name = 'Home'

urlpatterns = [
    path('', views.land_page, name='land_page'),
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup, name='signup'),
    path('market/', views.market, name='market'),
    path('profile', views.profile, name='profile'),
    path('live_graph', views.live_graph, name='live_graph'),
    path('predict_graph/', views.predict_graph, name="predict_graph"),
    path('error', views.error, name='error'),
    
]
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)