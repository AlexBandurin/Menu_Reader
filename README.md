# Menu_Reader

This is a web application that converts an image file of a restaurant menu into text using Optical Character Recognition (OCR). 
That text is then sent through a Machine Learning (ML) model to output a list of menu items using classification and Natural Language Processing techniques. 
The list is then passed to another ML model to be sorted into categories.

[View the App](https://menu-reader-1ada6a994a40.herokuapp.com/)

## Python Packages used:

### EasyOCR
An **optical character recognition** (OCR) package for converting an image file into distinct pieces of text (textboxes).
EasyOCR implements a deep learning execution based on Pytorch.
(https://github.com/JaidedAI/EasyOCR)
### Pandas
For organizing text provided by easyocr into a table (dataframe) and **feature engineering**. 
23 Menus have been collected from the internet, sent through ocr, and organized into a table, totaling **over 2500 rows** of text. 
### BERT (from Transformers)
Text was converted into **word embeddings** (768 features) to assist with classification modeling.  
### Sklearn and Xgboost
Each row of text has been labeled as a "menu item" or not (**binary classification**). The following algorithms were tested: Decision Tree (R^2 = 0.858), 
Random Forest (R^2 = 0.860), and **XGBoost (R^2 = 0.922)**, out of which XGBoost was deemed the most accurate. 

The menu items generated by this model have been sent through another model that relies on BERT as well as a few additional features that indicate the presense of key words within the menu to categorize these items. 

### Azure Cloud and Web Application

To make this application accessible to any user on the web, it is hosted on Heroku, a cloud platform ((https://heroku.com/)). Flask was used to develop this application.
[Web App Files](https://github.com/AlexBandurin/Menu_Reader/tree/master/menu_reader_app)
The algorithms used for OCR and text classification/categorization are quite memory intesive. To offload these computations, I used Azure Function App.
The image selected by a user is sent to the Function App in a JSON format (along with other informatin like image height and width), via RESTful API interface. 
The result (consisting of generated menu items sorted into categories) is returned to the web application on which it is displayed. 
[Function App Files](https://github.com/AlexBandurin/Menu_Reader/tree/master/menu_function)


<!---
The following features were used for classification:
- Text (word embeddings)
- width (width of text box)
- height (height of text box)
- uppercase (number of uppercase characters present in text)
- chars (number of characters)
- words (number of words)
- periods (number of periods)
- period_btw_numbers (count of periods between 2 numerical characters)
- number_end (count of numerical character at the end of text)
- numbers (number of numerical characters)
- commas (number of commas)
- exclamation (number of exclamation marks)
- question (number of numerical characters)
- colons (number of colons)
- underscores (number of underscores)
- dollar (number of dollar signs)
- punctuation (number of other punctuation characters)
- 2_periods_cnt	Item (count of 2 consecutive periods)
<br></br>
-->

