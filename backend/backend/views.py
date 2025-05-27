from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

# Multithreaded sim code can be run directly.

class HelloWorldView(APIView):
    def get(self, request):
    
        return Response({"message": "Hello, World!"}, status=status.HTTP_200_OK)
    

    def post(self, request):
        data = request.data
        return Response({"message": "Data received", "data": data}, status=status.HTTP_201_CREATED)