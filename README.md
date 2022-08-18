# deep-learning-challenge

# Report on the Neural Network Model
### Deep Learning Charity Funding Outcome Predictor Project using hyper-tuned Neural Networks.

## Overview:

I've created a tool for the  nonprofit foundation Alphabet Soup that can help it select applicants for funding with the best chance of success in their ventures. Using my knowledge of  machine learning and neural networks, I have used the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup. We were set a target of 75% accuracy for our model.
From Alphabet Soup’s business team, I received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

* **EIN** and **NAME**—Identification columns
* **APPLICATION_TYPE**—Alphabet Soup application type
* **AFFILIATION**—Affiliated sector of industry
* **CLASSIFICATION**—Government organization classification
* **USE_CASE**—Use case for funding
* **ORGANIZATION**—Organization type
* **STATUS**—Active status
* **INCOME_AMT**—Income classification
* **SPECIAL_CONSIDERATIONS**—Special consideration for application
* **ASK_AMT**—Funding amount requested
* **IS_SUCCESSFUL**—Was the money used effectively


# Steps Taken:

### 1: Data Preprocessing
* Dataset was checked for null and duplicated values
![Screenshot 2022-08-06 at 22 20 24](https://user-images.githubusercontent.com/67019030/183266401-f3e86c8b-d0e3-4b15-98d0-bc14dc4028e4.png)

* **EIN** and **NAME**—Identification columns removed from the input data because they are neither targets nor features
![Screenshot 2022-08-06 at 22 21 08](https://user-images.githubusercontent.com/67019030/183266418-f6677845-cc68-4ef9-99d1-d9e09415aee3.png)


* Created cutoff point to bin "rare" categorical variables together in a new value, `Other` for both `CLASSIFICATION` and `APPLICATION_TYPE`
![Screenshot 2022-08-06 at 22 32 50](https://user-images.githubusercontent.com/67019030/183266679-98f07d94-9993-4f99-8df4-ee16de08afe8.png)
![Screenshot 2022-08-06 at 22 35 47](https://user-images.githubusercontent.com/67019030/183266780-d2f9f084-6334-42c3-bad9-18ddb3f3b0df.png)

* Converted categorical data to numeric with `pd.get_dummies`, split the preprocessed data into features and target arrays, then lastly split into training and tesing datasets
![Screenshot 2022-08-06 at 22 40 32](https://user-images.githubusercontent.com/67019030/183266894-4e63a10c-327b-465e-bcb4-3528bd1a5c30.png)


Target Variable for the model: 
* **IS_SUCCESSFUL**

Feature Variables for the model: 
* **APPLICATION_TYPE**
* **AFFILIATION**
* **CLASSIFICATION**
* **USE_CASE**
* **ORGANIZATION**
* **STATUS**
* **INCOME_AMT**
* **SPECIAL_CONSIDERATIONS**
* **ASK_AMT**






### 2: Compiling, Training, and Evaluating the Model

I build the first model with the following parameters with low computation time in mind: 
* 2 hidden layers with 80, 30 neurons split (the input (node) feature was 43, 80 was chosen as the first layer as it is almost double the input_feature). With an hidden layer activation function of `relu` as this our go to for first model.
* Output node is 1 as it was binary classifier model with only one output: was the funding application succesfull yes or no. And an output layer activation of `sigmoid` as the model output is binary classification between 0 and 1.

I then increased the hidden layers to 3 and set the third hidden layer at 30 as the model prediction accuracy was below 75%:
![Screenshot 2022-08-06 at 23 05 08](https://user-images.githubusercontent.com/67019030/183267456-7b258bf3-6a46-40d0-a971-f435bd0d1973.png)


For the second model I decided to use `tanh` activation and 3 hidden layers with 90, 30, 20 neurons split and a `sigmoid` activation for output as the output doesn't change.
![Screenshot 2022-08-06 at 23 06 17](https://user-images.githubusercontent.com/67019030/183267476-bcfa5db1-e9e1-48af-8472-f53f24344373.png)


I experimented with increasing nodes and neurons, with changing other parameters to get a better accuracy but despite doing this both models came below the 75% threshold.


### 3: Optimize the Model

I decided to use an automated model optimizer to get the most accurate model possible by creating method that creates a `keras` Sequential model using the `keras-tuner library` with hyperparametes options. 
![Screenshot 2022-08-06 at 23 10 28](https://user-images.githubusercontent.com/67019030/183267573-22a806a6-8e10-4b08-bd50-70a9533866a1.png)
Which will automatically tune the hyperpyrameters until it gets the most accurate model.
![Screenshot 2022-08-06 at 20 05 12](https://user-images.githubusercontent.com/67019030/183267609-9d4e0d27-2df0-49c9-81ee-4e4012978067.png)
 
 * The best model from the keras tuner method achieved 73% prediction accuracy using a sigmoid activation function with input node of 46, 6 hidden layers at a 51, 81, 71, 6, 41, 91 neurons split and 100 training epochs.
![Screenshot 2022-08-06 at 23 17 09](https://user-images.githubusercontent.com/67019030/183267670-219340f3-b526-4971-bbf2-9c1ab4349fc9.png)

# Further and Final Optimization
I kept the `Name` column for my final Optimized Model as I still hadn't reached the goal of 75% accuracy. Kepping the keras-tuner the same apart from lowering the epochs from 100 to 50 for time optimization.

![Screenshot 2022-08-18 at 21 14 01](https://user-images.githubusercontent.com/67019030/185486209-afd1c3dd-cea3-49e6-84ef-3dbfb97aef80.png)


## Final and best optimized model achieved 80% accuracy , which exceeds the 75% goal and is our best model.

![Screenshot 2022-08-18 at 20 57 31](https://user-images.githubusercontent.com/67019030/185486450-61a6cc03-eed0-4cec-a628-095438c5590a.png)

### All Top 3 Models were around the 80%.
All had a sigmoid activation with different number of input nodes and hidden layers:

![Screenshot 2022-08-18 at 20 57 10](https://user-images.githubusercontent.com/67019030/185484546-0086d851-a10a-4c6d-b65f-7ab2cf923b0e.png)


# Summary: 
The final automatically optimized neural network trained model from the keras tuner method achieved 80% prediction accuracy with a 0.45 loss, using a sigmoid activation function with input node of 76; 5 hidden layers at a 16, 21, 26, 11, 21, neurons split and 50 training epochs.  Performing better than the non automized model.
Keeping the `Name` column was crucial in achieving and and going beyond the target. This shows the importance of the shape of your datasets before you preprocess it. 
