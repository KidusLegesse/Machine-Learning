# Machine-Learning :computer: :books: :crystal_ball:


In this program I implement two machine learning models (neural network & KNN) and utilize a Cleveland database on heart disease to train the models (Neural Network & KNN) in order to predict whether or not a patient has Heart Disease.

##Information about how I used the data:
    The Cleaveland database[https://doi.org/10.24432/C52P4X] has 76 attributes however most experiments use 14 of them. My Neural network and KNN models are trained on the data with the most commonly used 14 attributes. It gives a binary classification (Heart Disease present or not present). After testing the models and tweaking the parameters I got on average an accuracy of 91% for the Neural Network and 89% for KNN.

    ### :zap:The atributes are:
    
        1.Age in years      
        2.Sex  
        3.Chest pain type 
            1 if typical angina
            2 if atypical angina
            3 if non-anginal pain
            4 if asymptomatic       
        4.Resting blood pressure (in mm Hg on admission to the hospital) 
        5.Serum cholesterol in mg/dl  
        6.Fasting blood sugar > 120 mg/dl       
        7.Resting ECG results  
        8.Maximum heart rate achieved   
        9.Exercise induced angina     
        10.ST depression induced by exercise relative to rest  
        11.The slope of the peak exercise ST segment     
        12.Number of major vessels (0-3) colored by flourosopy       
        13.Thal   
        14.**Target Value** diagnosis of heart disease (angiographic disease status)
            Target value = 
            0 if < 50% diameter narrowing
            otherwise 1 


:o:The target values are [0,1,2,3,4]. 0(min-value) corresponds to no presence of heart disease in a patient and 4(max-value) corresponds to the highest level of certainty of the presence of heart disease.


:space_invader: I changed this multiclass classification into a binary classification because of how the data is used in experiments.

>"Experiments with the Cleveland database have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0)."

`new_targets = np.where(old_targets != 0, 1, old_targets)`

:star2:There were not too many missing values in the original dataset however in order to get rid of the few there were I first had to convert them to NaN because they were marked as (?).
The following code demonstrates how I did this:

`dataSet.replace('?', np.nan, inplace=True)`

:snowman:Then I replaced all the NaN values with the mean of each column using SimpleImputer. I chose to replace them with the mean since it would be the best guess as to what the values would be.
The following code demonstrates how I did this:   

`replace_missing= SimpleImputer(strategy='mean')`
`clean_dataset = pd.DataFrame(replace_missing.fit_transform(dataSet), columns=dataSet.columns)`

## :chart_with_upwards_trend:Results:


## Tools Used:

    Tensorflow, Matplotlib, Sklearn, Mumpy, Pandas, Imblearn


## *Citation*:
*Janosi,Andras, Steinbrunn,William, Pfisterer,Matthias, and Detrano,Robert. (1988). Heart Disease. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X.*