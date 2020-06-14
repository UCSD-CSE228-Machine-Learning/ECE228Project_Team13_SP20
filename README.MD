# ECE228Project_Team13_SP20

### Dataset "UrbanSound8K" link: wget https://goo.gl/8hY5ER


## File Organization 
### dataset/ ----------- Contains dataset API.<br>
It contains a dataset.py file that defines the dataset API. The dataset class can automatically load, preprocess and generate data batch for training and testing.<br>



### Deep Learning/ ------- Contains results of deep machine learning methods.<br>
The deep learning package contain the ipynb files of RNN, CNN, DNN, and resnet18, it uses the pytorch network models and provides the Accuracy and loss. The ResNet implementation is in the datasets package.

### Traditional/ ------- Contains results of traditional machine learning methods.<br>
There are four files in the "traditional" fold -- two for mel spectrogram of lower dimension and two for mfcc of lower dimension. Each file applies four traditonal methods which are SVM, KNN, RF and XGBoost and has only one fold results as demo instead of taking 10-fold cross validation.

