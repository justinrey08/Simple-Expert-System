# Let's first import the necessary packages
# For our data analysis and wrangling
import pandas as pd
# for our GUI
from tkinter import *
import PIL
from PIL import ImageTk
from PIL import Image
# For our data visualization
import seaborn as sns
# lastly, for our machine learning
# always hoist your valuables
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics

ty = [[]]
temp_arr = []
root = Tk()
root.iconbitmap('icon.ico')
root.title("Breast Cancer Predictor Expert Systems")
# Let's create the input fields for the necessary data.
fst = Entry(root, width=50, borderwidth=5)

snd = Entry(root, width=50, borderwidth=5)

trd = Entry(root, width=50, borderwidth=5)

fth = Entry(root, width=50, borderwidth=5)

ffth = Entry(root, width=50, borderwidth=5)

sxth = Entry(root, width=50, borderwidth=5)

svth = Entry(root, width=50, borderwidth=5)

eth = Entry(root, width=50, borderwidth=5)

# Let's put some labels within the corresponding input fields
mse = Label(root, text="Please enter the following data needed: ")
mse.grid(row=0, column=1)
fstl = Label(root, text="Radius_Mean")
sndl = Label(root, text="Perimeter_Mean")
trdl = Label(root, text="Area_Mean")
fthl = Label(root, text="Radius_Se")
ffthl = Label(root, text="Perimeter_Se")
sxthl = Label(root, text="Radius_Worst")
svthl = Label(root, text="Perimeter_Worst")
ethl = Label(root, text="Area_Worst")

# Let's position the input fields and labels
fst.grid(row=1, column=1)
snd.grid(row=2, column=1)
trd.grid(row=3, column=1)
fth.grid(row=4, column=1)
ffth.grid(row=5, column=1)
sxth.grid(row=6, column=1)
svth.grid(row=7, column=1)
eth.grid(row=8, column=1)
fstl.grid(row=1, column=0)
sndl.grid(row=2, column=0)
trdl.grid(row=3, column=0)
fthl.grid(row=4, column=0)
ffthl.grid(row=5, column=0)
sxthl.grid(row=6, column=0)
svthl.grid(row=7, column=0)
ethl.grid(row=8, column=0)


def Malwind():
    global img
    maltop = Toplevel()
    maltop.iconbitmap('icon.ico')
    maltop.title("Expert Systems Diagnosis Results")
    mgt1 = Label(maltop, text="The breast lump is diagnosed to be: Malignant").grid(row=5, column=2)
    mgt2 = Label(maltop, text="The following symptoms could be observed:").grid(row=6, column=2)
    mgt3 = Label(maltop, text="Skin irritation\nPain or tenderness of the nipple\nBloody nipple discharge").grid(row=7, column=2)
    canvas = Canvas(maltop, width = 80, height = 80)
    canvas.grid(row = 1, column = 2)
    img = ImageTk.PhotoImage(Image.open("softer.png"))
    canvas.create_image(2, 2, anchor = NW, image = img)

def Belwind():
    global img
    maltop = Toplevel()
    maltop.iconbitmap('icon.ico')
    maltop.title("Expert Systems Diagnosis Results")
    mgt1 = Label(maltop, text="The breast lump is diagnosed to be: Benign").grid(row=5, column=2)
    mgt2 = Label(maltop, text="The following symptoms could be observed:").grid(row=6, column=2)
    mgt3 = Label(maltop, text="Nipple pain or retraction\nPain or tenderness of the nipple\nDischarge from the breast that is not milk").grid(row=7, column=2)
    canvas = Canvas(maltop, width=80, height=80)
    canvas.grid(row=1, column=2)
    img = ImageTk.PhotoImage(Image.open("softer.png"))
    canvas.create_image(2, 2, anchor=NW, image=img)

# Now, we create the model
# Let us now import our data and take a look

# always use relative file paths
data = pd.read_csv(r"data.csv")
print(data.head())
print(data.tail())
# We then also check for null values.
data.isnull().sum()
# and drp them
data.drop("Unnamed: 32", axis=1, inplace=True)

#let us make the variables for each list that we need
features_mean = list(data.columns[2:12])
features_se = list(data.columns[12:22])
features_worst = list(data.columns[22:32])
# print(features_mean)
# print(features_se)
# print(features_worst)

# Now let's drop the columns that have correlation and high correlation.
drplist1 = ['id', 'diagnosis', 'perimeter_mean', 'radius_mean', 'compactness_mean', 'concave points_mean',
            'radius_se', 'perimeter_se', 'radius_worst', 'perimeter_worst',
            'compactness_worst', 'concave points_worst', 'compactness_se',
            'concave points_se', 'texture_worst', 'area_worst']
x1 = data.drop(drplist1, axis=1)

# Now we do the modelling.
y = data.diagnosis

# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.3, random_state=42)

# random forest classifier with n_estimators=10 (default)
rfne = RandomForestClassifier(random_state=43)
rfne = rfne.fit(x_train, y_train)

ac = accuracy_score(y_test, rfne.predict(x_test))
print('Accuracy is: ', ac)
cm = confusion_matrix(y_test, rfne.predict(x_test))
hm = sns.heatmap(cm, annot=True, fmt="d")
hm.get_ylim()
hm.set_ylim(2.0, 0)


def parseData(array):
    # We will use these variables for our prediction
    pred_var = ['radius_mean', 'perimeter_mean', 'area_mean', 'radius_se', 'perimeter_se', 'radius_worst',
                'perimeter_worst', 'area_worst']
    train, test = train_test_split(data, test_size=0.3)  # in this our main data is split into train and test
    train_X = train[pred_var]  # taking the training data input
    train_y = train.diagnosis  # This is output of our training data

    # https://medium.com/@hjhuney/implementing-a-random-forest-classification-model-in-python-583891c99652

    test_X = test[pred_var]  # taking test data inputs
    test_y = test.diagnosis  # output value of test data

    print(f"TestX: {test_X}")
    rfcmodel = RandomForestClassifier(n_estimators=100)  # let's try it with a simple random forest model
    rfcmodel.fit(train_X, train_y)  # now fit our model for training data
    prediction = rfcmodel.predict(test_X)  # predict for the test data

    # pass it as an array instead
    # you're passing a 1 dimensional array, pass a 2 dimensional array instead... this is up to you Mr. Data Science
    yow = rfcmodel.predict(array)
    print(yow)
    print(f'Accuracy Check: {metrics.accuracy_score(prediction, test_y)}')  # to check the accuracy
    # here we will use accuracy measurement between our predicted value and our test output values
    return yow


def enterCredata(ty):
    print(f"Expected Output: {type(ty[0])}")
    if ty[0] == "M":
        Malwind()
    else:
        Belwind()

def myClick():
    # parseData()
    # enterCredata()
    # basically loops through all the input boxes whilst skipping the empty ones to prevent ValueError
    for x in root.winfo_children():
        if x.winfo_class() == 'Entry':
            if x.get() != '':
                # push all to the array
                print(x.get())
                temp_arr.append(float(x.get()))

    ty[0] = temp_arr
    print(ty)
    enterCredata(parseData(ty))

button1 = Button(root, text='Enter Credentials', command=myClick)
button1.grid(row=9, column=1)

# run the GUI
root.mainloop()