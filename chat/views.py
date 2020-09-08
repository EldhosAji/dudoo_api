from .models import chat_index
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .ai.Chat import ai_chat

@api_view(['GET'])
def aiResponse(request):
    q = request.GET.get("q").replace("+"," ")

    response = ai_chat(q)
    if response =="Sorry, i don't understand" and not chat_index.objects.filter(query=q).exists():
        data = chat_index(query=q)
        data.save()

    return Response(response)
