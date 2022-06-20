from attr import field
from django import forms
from .models import PytorchModel 

class PytorchForm(forms.ModelForm):
    class Meta:
        model = PytorchModel
        fields = ['title']