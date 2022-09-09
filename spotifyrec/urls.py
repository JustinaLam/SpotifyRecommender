from django.urls import path
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from spotifyrec import views
from spotifyrec.models import SongRec

recs_list_view = views.RecsListView.as_view(
    queryset=SongRec.objects.order_by("-similarity")[:25],  # order by dec sim, :25 limits results to 25 highest sim
    context_object_name="recs_list",
    template_name="spotifyrec/recsList.html",
)

urlpatterns = [
    path("", views.home, name="default"),
    path("spotifyrec/home", views.home, name="home"),
    path("spotifyrec/getRecsList/<playlistURL>", views.list_recs, name="getRecsList"),
    path("spotifyrec/showRecsList/<x>", recs_list_view, name="showRecsList"),
    path("spotifyrec/request", views.request_page, name="request"),
]

urlpatterns += staticfiles_urlpatterns()

