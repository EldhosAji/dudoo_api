from django.db import models

class chat_index(models.Model):
    query = models.CharField(max_length=1000,primary_key = True)
    def __str__(self):
        return self.query
