from django.db import models

# Create your models here.
from django.db import models

class SongRec(models.Model):
    track = models.TextField(default="")
    trackURI = models.TextField(default="")
    artist = models.TextField(default="")
    album = models.TextField(default="")
    similarity = models.FloatField(default=0.0)

    def __str__(self):
        """Returns a string representation of a song rec."""
        return f"'{self.track}' by '{self.artist}' in album '{self.album}' with similarity '{self.similarity}'"
