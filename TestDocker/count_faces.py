import cv2 
def count_faces(image_path): 
    
  # Load the Haar cascade classifier. 
  face_cascade = cv2.CascadeClassifier('C:/Users/leemn/Downloads/haarcascade.xml') 
    
  # Read the image. 
  image = cv2.imread(image_path) 
    
  # Convert the image to grayscale. 
  grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
  # Detect faces in the image. 
  faces = face_cascade.detectMultiScale( 
      grayscale_image, 
      scaleFactor=1.1, 
      minNeighbors=5, 
      minSize=(30, 30)) 
    
  # Count the number of faces. 
  number_of_faces = len(faces) 
  return number_of_faces 
if __name__ == '__main__': 
   
# The path to the image file. 
  image_path = "C:/Users/leemn/Downloads/image.jpg"
  
 # Count the number of faces in the image. 
  number_of_faces = count_faces(image_path) 
   
# Print the number of faces. 
  print(number_of_faces)