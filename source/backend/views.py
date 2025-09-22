from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.views.generic import TemplateView

import plotly.express as px
from plotly.offline import plot
from plotly.graph_objs import Figure, Scatter

# Multithreaded sim code can be run directly.

from simulation import runSimulationPool

import numpy as np
import pandas as pd



class HelloWorldView(APIView):
    def get(self, request):
        #print(runSimulationPool())
        return Response({"message": "Hello, World!"}, status=status.HTTP_200_OK)
    

    def post(self, request):
        data = request.data
        return Response({"message": "Data received", "data": data}, status=status.HTTP_201_CREATED)
    

def ternaryTestPlot(results):
    fig = px.line_ternary(results, a="R", b="P", c="S", title="RPS Moran Process Trajectory", labels={"R":"Rock", "P":"Paper", "S":"Scissors"}, width=500)
    
    plot_div = plot(fig, output_type='div')
    

    return plot_div



class SimulationDataView(APIView):

    template_name = "landing.html"
    def post(self, request):
        try:
            results = runSimulationPool(
                simulations=1,
                popSize=10000,
                initialDist=[0.7,0.1,0.1,0.1],
                iterations=2000000,
                w=0.2,
                H=3)
            response = [r.tolist() for r in results]

            return Response({"results": response}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"Simulation failed ": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


    

class SimpleTemplateView(TemplateView):
    template_name = "landing.html"

    rpsArray = np.array([[0, -1 , 1], [1, 0, -1], [-1, 1, 0]])
  

    def post(self, request, *args, **kwargs):

        results = runSimulationPool(
            matrix=self.rpsArray,
            simulations=1,
            popSize=30000,
            initialDist=[0.5, 0.25,0.25],
            iterations=100000,
            w=0.2,
            H=2)
        
        df_RPS_MO = pd.DataFrame({"R": results[0][0], "P": results[0][1], "S": results[0][2]})
  

        plot_div = ternaryTestPlot(df_RPS_MO)

        context = self.get_context_data(plots=[plot_div])
        return self.render_to_response(context)

    def get_context_data(self, **kwargs):


        context = super().get_context_data(**kwargs)
        context['title'] = "First view" 
        context['plots'] = kwargs.get('plots', None)
        context['plots'] = kwargs.get('plots', None)
        return context
    