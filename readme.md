# Classification Exam FISI-3650

Welcome to the classification exam for the entrepreneurship course at Universidad de Los Andes. Your goal is to create a machine learning model able to identify varieties of rice from images. 

The challenge contains a directory called **Rice_Image_Dataset** which includes training and testing data:
- The training data is contained into the **Train** folder, this folder contains a folder with images for each class.
- The testing data can be found into the **Test** folder, this folder contains several images named after it class. The class of each image will be the initial string of the label. 

<p align="center">
  <img src="./Rice_Image_Dataset/Test/Arborio (10001).jpg" />
</p>

## Context 
------
Computer vision is one of the fields that take more advantage of machine learning, given that programing a series of criteria to classify an image by hand is quite challenging. This is because real world images present high variability due to variables like light conditions, perspective, camera settings, signal shapes, among others. Being able to build a model from scracht and understand why does it work, is a fundamental skill for a machine learning enginering. Therefore, we will be able to test your knowledge of the field by seeing the quality of your code and challenge your understanding of your own code. 

There are multiple methods to work with computer vision such as [neural networks](https://www.investopedia.com/terms/n/neuralnetwork.asp#:~:text=A%20neural%20network%20is%20a,organic%20or%20artificial%20in%20nature), [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) and [support vector machines](https://en.wikipedia.org/wiki/Support-vector_machine). We strongly suggest using neural networks to solve the challenge, as we have saw better perfomance using an specific architecture of this method. However, we love seeing original approaches with good perfomance. 

## Deliverable
------
Your deliverable is a python script that must be called **model.py**, this file must have a class named Model that have a **predict(file_path)** method. The class will contain your pre-trained model, and must load an image from any size specified in the **file_path** and return an integer indicating the predicted class for that image. 

Your model can be written in pytorch, tensorflow, scikit-learn or be written entirely by you. Other libraries will not be taken into account as your model will be tested automatically. 

## Evaluation
-----
We will evaluate your code accuraccy against a larger test dataset. The accuracy will measure how many classes where identified as correct with respect to the total number of files. The **Test** dataset use for evaluation has 25000 images compared to the one provided in the repository which size has 10000 images. Any code that takes longer than 27 min to evalute the entire dataset will be killed and the accuracy will be computed with the images predicted in less than 27 minutes.

This repository includes an **eval.py** script that will give an accuracy score (higher is better). You can use it to test your solutions against the labeled examples. We will use this same script to evaluate your solution.

### **We will grade solution based on the mean accuracy of the entire submissions, the minimum accuracy needed to be accepted in the course is 65%.** 

Hints
------
- We have computed result with accuracies higher than 90% on the **Train** set. You can aim to get results with this score. 
- The highest accuracy we have reached for the **Test** is 83.372%.
- Normalize all the images before parsing into your model, images can have different shapes.
- Use the three color channels to parse info into your model.
- The training dataset is well balanced, equally sized and the images are highly detailed. Therefore we advice to spend more time on the model rather than preprocessing it. 
- We have saw that the denser the model the better, however balance it to not spend to much time on the evaluation. Remember **27 mins** is the max. 
- Check that your code runs on the eval script, if it doesn't run we will not be able to give you a grade. Include the libraries you need to run the code. 

[Application form](https://forms.office.com/pages/responsepage.aspx?id=fAS9-kj_KkmLu4-YufucyuvPP8FxoDxPtQnJHZ3zr3NURUdCNUU3T1o1RkRMUUg3RkxURk9LMjdFRi4u)
-------
Use the link to pre-signed for the course, we will have more communication for the ones signed in. 