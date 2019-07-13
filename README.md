# 1. Data Preprocessing 
## 1.1 Drop Unnecessary Columns
The ID column is not fed to the ML Model. The UsageClass is "Physical", CheckoutType is "Horizon", CheckoutYear is "2005", and the CheckoutMonth is "4" for the full training data. Number of Checkouts don't have affect on the output MaterialType Variable. Creator columns has too many NAN to consider.
## 1.2 Necessary Columns
Now we are left with three important columns for input variables : "Title", "Subjects", and "Publisher".
## 1.3 Handling Text Columns
There are many ways of handling text columns but here I use TfidfVectorizer with a text_process function that removes all punctuations, stopwords and returns a cleaned text as list of words.
We can create 3 type of preprocessings:
	 a. With only "Title" column
	 b. With "Title" and "Subject" columns
	 c. With "Title", "Subject" and "Publisher" column
## 1.4 Label Encoding Output Variable
We would have to LabelEncode the output variable as it has multiple categories.

# 2. Training
## 2.1 Choosing the right model
We define a fitpred() function that prints out the confusion matrix, accuracy and classification report of all the models.
Models trained are : nb = MultinomialNB(),BernoulliNB(),svm.SVC(gamma='scale'),xgb.XGBClassifier(),DecisionTreeClassifier(),RandomForestClassifier().
After comparing these models we get that XGBClassifier gives the best result.
## 2.2 Choosing the right subset of columns :
We train the XGBClassifier over all the three preprocessing categories and get the following scores:
	 a. With only "Title" column - 67.729%
	 b. With "Title" and "Subject" columns - 66.479%
	 c. With "Title", "Subject" and "Publisher" column - 55.44%
# 3. Conclusion
	We use XGBClassifier with only "Title" Column as the input and "MaterialType" as the output.
