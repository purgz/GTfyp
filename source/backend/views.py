from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.views.generic import TemplateView

# Multithreaded sim code can be run directly.

from simulation import runSimulationPool

class HelloWorldView(APIView):
    def get(self, request):
        #print(runSimulationPool())
        return Response({"message": "Hello, World!"}, status=status.HTTP_200_OK)
    

    def post(self, request):
        data = request.data
        return Response({"message": "Data received", "data": data}, status=status.HTTP_201_CREATED)
    


class SimpleTemplateView(TemplateView):
    template_name = "landing.html"


    def post(self, request, *args, **kwargs):
        results = runSimulationPool()
        context = self.get_context_data(results=results)
        return self.render_to_response(context)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = "First view" 
        return context
    