from django.shortcuts import render, redirect
from .models import PytorchModel 
from .forms import PytorchForm
from .recognition.recognize import main


# Create your views here.
def upload(request):
    if request.method == 'POST':
        form = PytorchForm(request.POST)

        if form.is_valid():
            form.save()
            # text = PytorchModel.objects.get(pk=PytorchModel.objects.count())
            # output = main(text.title)

            # text.first_text = output[0]
            # text.save()

            return redirect('pytorchapp:result')
    else:
        form = PytorchForm()
    
    context = {'form': form}
    return render(request, '../templates/upload.html', context)


def result(request):
    # texts = PytorchModel.objects.all().order_by('-pk')
    # print(texts)
    # context = {'textBody': texts[0]}

    text = PytorchModel.objects.get(pk=PytorchModel.objects.count())
    output = main(text.title)

    textAns = {'textBody': output[0]}
    
    return render(request, '../templates/result.html', textAns)