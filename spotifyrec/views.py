from django.shortcuts import render
from django.shortcuts import redirect
from django.views.generic import ListView

from spotifyrec.models import SongRec
import spotifyrec.spotifyRec

import pandas as pd

class RecsListView(ListView):
    """Renders the home page, with a list of all messages."""
    model = SongRec

    def get_context_data(self, **kwargs):
        context = super(RecsListView, self).get_context_data(**kwargs)
        return context
        

def home(request):
    return render(request, 'spotifyrec/index.html')


def request_page(request):
    if (request.GET.get('generateBtn')):
        return redirect("getRecsList", request.GET.get('inputURL'))
        
    else:
        return redirect("home")

def list_recs(request, playlistURL):
    SongRec.objects.all().delete()
    # Run script to generate song recommendations as SongRec objects
    spotifyrec.spotifyRec.init(playlistURL)
    
    recs_list_view = RecsListView.as_view(
        queryset=SongRec.objects.order_by("-similarity")[:25],  # order by dec sim, :25 limits results to 25 highest sim
        context_object_name="recs_list",
        template_name="spotifyrec/recsList.html",
    )
    return redirect("showRecsList", recs_list_view)

