List of files-

1.Dataset- Contains three folders-
Sample-Contains images in three folders(one folder for each class) for testing the model on sample data
Train- Contains images in three folders(one folder for each class) for training the model 
Test- Contains images in three folders(one folder for each class) for tresting the model in three folders

2.Model- Contains the various models-
 part1.pth-model for iteration 1
 part1_with_kFold.pth-Iteration 1 model with k-fold
 modified.pth- Model after change in architecture
 latest.pth- the final model

3.Output- Contains images used to test the model on data present in folder Dataset/Sample

4.DatsetInfo.docx- Information about the dataset

5.train.py- python code to train and test the model

6.train-Part1.py- script for iteration 1

7.train-Part1_with_kFOld.py- to run k-fold on iteration 1

8.modified.py- after change in architecture

9.test.py- pyhton code to test sample data

10.Report.pdf- Report of the project

11.Expectation of Orginality form

STEPS FOR-

a)Training the model-

1.Make sure to have the Dataset folder along with all the subfolders and files present in the same directory as that of train.py
2.Make sure to have the Model folder present in the same directory as that of train.py to save the model.
3.Execute train.py 
4.Model with name mask_detection-cnn.pth would be created in Model folder after successful execution.

b) Testing on Sample data
1.Make sure to have the Dataset folder along with subfolder Sample in the same directory as that of test.py
2.Make sure to have the Output folder present in the same directory as that of test.py to save the prediction of each image.
3.Execute test.py, on successful execution, all the analysis would be present on the console, you can also check predicted class for each image.
4.To check predicted class for each image,you will find each image with predicted class mentioned in folder Output.
Each image would have prediction mentioned on it in the format
[predicted class]:[True/False](True if the prediction is correct otherwise False). For example- "Mask:True" means the predicted class for the image is Mask and it is correctly predicted.
"NotPerson:False" means the predicted class for the image is NotPerson and the prediction is wrong.
5. You will need to zoom in to see the image as the image size is quite small.
