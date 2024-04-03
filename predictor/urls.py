
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('', include(('Home.urls', 'Home'), namespace='Home')),
    path('admin/', admin.site.urls),
]
