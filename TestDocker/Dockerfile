FROM python:3.8-slim-buster 
WORKDIR /app 
COPY . . 
  
RUN pip install opencv-python-headless==4.5.3.56
CMD ["python", "count_faces.py"] 