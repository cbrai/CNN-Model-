# CNN-Model-
Binary Classifier for Birds and Drones
Be ready with your libraries installed which I have already mentioned in requirements.txt file

# Model Building
Steps to create your own Model
Step1: Create two folders of each class i.e. (in my case it was birds and drones) but you can create any class,
Step2: Create two folders named as training and validation and add photos in 9:1 ratio in both the folders of each class

# So basically your directories should look like 
root_folder/training/birds/imgaes
root_folder/training/drones/imgaes
Step3: Now train the model 
Step4: And then save the model in .h5 format for further use in deployement.


# Deployement
Step1: Keep the folders as it is I have in my repository
Step2: After running the app.py file a flask_ngrok link will be generated which is active for 2 hours 
Step3: you can upload any images of any two classes and check your model against prediction

# Note
My model is trained for 128 x 128 size which can be changed in the model.py code and the same has to be chnaged in app.py file for different size
