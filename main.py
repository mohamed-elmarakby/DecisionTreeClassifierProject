from __future__ import print_function
import pandas as pd
import datetime
from tkinter import *
from os import *
from tkinter import filedialog

root = Tk()
root.title('Decision Tree Classifier')
lv = Listbox(root,width=100)
trainFolder = ''
devFolder = ''
testFolder = ''
savedDevLocation = ''
savedTestLocation = ''


def saveDevLoc():
    root.savedLocation1 = root.filedialog.askdirectory()
    global savedDevLocation
    savedDevLocation = root.savedLocation1


def saveTesLoc():
    root.savedLocation2 = root.filedialog.askdirectory()
    global savedTestLocation
    savedTestLocation = root.savedLocation2


def openTrain():
    root.train = filedialog.askopenfilename(initialdir="", title="Select Train File",
                                            filetypes=(("CSV Files", "*.csv"), ("Choose only CSV", "*.*")))
    global trainFolder
    trainFolder = root.train


def openDev():
    root.dev = filedialog.askopenfilename(initialdir="", title="Select Dev File",
                                          filetypes=(("CSV Files", "*.csv"), ("Choose only CSV", "*.*")))
    global devFolder
    devFolder = root.dev


def openTest():
    root.test = filedialog.askopenfilename(initialdir="", title="Select Test File",
                                           filetypes=(("CSV Files", "*.csv"), ("Choose only CSV", "*.*")))
    global testFolder
    testFolder = root.test


def clearLog():
    lv.delete(0,END)


def decisionTreeClassifier(type):
    global trainFolder
    global devFolder
    training_data = pd.read_csv(f'{trainFolder}')
    training_data = training_data.drop(columns=['reviews.text'])
    cols = training_data.columns
    trainedData = training_data.values.tolist()
    print('Training Begun...')
    lv.insert(END,f"Training Begun at: {datetime.datetime.now().strftime('%H:%M:%S')}")
    if type == 'dev':
        deving_data = pd.read_csv(f'{devFolder}')
        devedData = deving_data.values.tolist()
    elif type == 'test':
        testing_data = pd.read_csv(f'{testFolder}')
        testedData = testing_data.values.tolist()
    elif type == 'review_test':
        testing_data = pd.read_csv('reviews.csv')
        testedData = testing_data.values.tolist()

    def unique_vals(rows, col):
        return set([row[col] for row in rows])

    numberOfRows = int(training_data.shape[0])
    numberOfColumns = int(training_data.shape[1])

    def class_counts(rows):
        counts = {}
        for row in rows:
            rate = row[-1]
            if rate not in counts:
                counts[rate] = 0
            counts[rate] += 1
        return counts

    def is_numeric(value):
        return isinstance(value, int) or isinstance(value, float)

    class QestionToBeAsked:

        def __init__(self, numberOfColumn, searchedForValue):
            self.classColumn = numberOfColumn
            self.value = searchedForValue

        def isSame(self, example):
            # Compare the feature value in an example to the
            # feature value in this question.
            val = example[self.classColumn]
            if is_numeric(val):
                return val == self.value
            else:
                return val == self.value

    def partitioning(rows, question):
        true_rows, false_rows = [], []
        for row in rows:
            if question.isSame(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows

    def giniFunction(rows):
        counts = class_counts(rows)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(rows))
            impurity -= prob_of_lbl ** 2
        return impurity

    def infoGain(left, right, current_uncertainty):
        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * giniFunction(left) - (1 - p) * giniFunction(right)

    def findBestSplit(rows):
        maxGain = 0
        bestQuestion = None
        current_uncertainty = giniFunction(rows)
        n_features = len(rows[0]) - 1

        for col in range(n_features):

            values = set([row[col] for row in rows])

            for val in values:

                question = QestionToBeAsked(col, val)

                trueRowsHere, falseRowsHere = partitioning(rows, question)

                if len(trueRowsHere) == 0 or len(falseRowsHere) == 0:
                    continue

                informationGained = infoGain(trueRowsHere, falseRowsHere, current_uncertainty)

                if informationGained >= maxGain:
                    maxGain, bestQuestion = informationGained, question

        return maxGain, bestQuestion

    class Leaf:
        def __init__(self, rows):
            self.predictions = class_counts(rows)

    class DecisionNode:

        def __init__(self,
                     branchingQuestion,
                     trueDecisionBranch,
                     falseDecisionBranch):
            self.question = branchingQuestion
            self.true_branch = trueDecisionBranch
            self.false_branch = falseDecisionBranch

    def treeBuilding(rows):
        gain, question = findBestSplit(rows)

        if gain == 0:
            return Leaf(rows)

        true_rows, false_rows = partitioning(rows, question)

        true_branch = treeBuilding(true_rows)

        false_branch = treeBuilding(false_rows)

        return DecisionNode(question, true_branch, false_branch)

    ourTree = treeBuilding(trainedData)

    def classify(row, node):
        if isinstance(node, Leaf):
            return node.predictions

        if node.question.isSame(row):
            return classify(row, node.true_branch)
        else:
            return classify(row, node.false_branch)

    # def print_leaf(counts):
    #     total = sum(counts.values()) * 1.0
    #     probs = {}
    #     for lbl in counts.keys():
    #         probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    #     return probs
    print('Training End...')
    l = Label(root, text=f"Training End at: {datetime.datetime.now().strftime('%H:%M:%S')}")
    lv.insert(END, f"Training End at: {datetime.datetime.now().strftime('%H:%M:%S')}")
    if type == 'dev':
        rows = []
        i = 0
        for row in devedData:
            rows.append(classify(row, ourTree))
            # print("Review Number: %s,Actual: %s, Predicted: %s" %
            #       (i, row[-1],classify(row, ourTree)))
            i += 1

        y = 0
        for y in range(len(rows)):
            if 'Positive' in rows[y] and 'Negative' not in rows[y]:
                rows[y] = 'Positive'
            elif 'Negative' in rows[y] and 'Positive' not in rows[y]:
                rows[y] = 'Negative'
            else:
                if int(rows[y]['Positive']) > int(rows[y]['Negative']):
                    rows[y] = 'Positive'
                else:
                    rows[y] = 'Negative'
            y += 1

        df = pd.DataFrame(rows, columns=["rating"])
        df.to_csv('devList.csv', index=False)
        acc = pd.read_csv('devList.csv').values.tolist()
        testedResult = class_counts(acc)
        originalResult = class_counts(devedData)
        # print(f'original => {originalResult}')
        # print(f'tested => {testedResult}')
        accu = 100 * (1 - ((abs(int(originalResult['Positive']) - int(testedResult['Positive'])) + abs(
            int(originalResult['Negative']) - int(testedResult['Negative']))) / (
                                   int(originalResult['Positive']) + int(originalResult['Negative']))))
        lv.insert(END,f'Accuracy = {accu}%')
        lv.insert(END,f'Tested Results = {testedResult}')
        lv.insert(END,f'Original Results= {originalResult}')

    elif type == 'test' or type == 'review_test':
        rows = []
        i = 0
        for row in testedData:
            rows.append(classify(row, ourTree))
            # print("Review Number: %s, Predicted: %s" %
            #       (i, classify(row, ourTree)))
            i += 1
        y = 0
        for y in range(len(rows)):
            if 'Positive' in rows[y] and 'Negative' not in rows[y]:
                rows[y] = 'Positive'
            elif 'Negative' in rows[y] and 'Positive' not in rows[y]:
                rows[y] = 'Negative'
            else:
                if int(rows[y]['Positive']) > int(rows[y]['Negative']):
                    rows[y] = 'Positive'
                else:
                    rows[y] = 'Negative'
            y += 1
        df = pd.DataFrame(rows, columns=["rating"])
        if type == 'test':
            df.to_csv('testList.csv', index=False)
            acc = pd.read_csv('testList.csv').values.tolist()
            testedResult = class_counts(acc)
            # print(f'akno my3rfosh we by3ml 3leh test => {testedResult}')
            lv.insert(END,f'Test result = {testedResult}')
            lv.insert(END,f'Test results .csv file made at Path = {path.dirname(path.realpath(__file__))}')
            lv.insert(END,f'Name = testList.csv')
        elif type == 'review_test':
            df.to_csv('reviewResult.csv', index=False)
            acc = pd.read_csv('reviewResult.csv').values.tolist()
            testedResult = class_counts(acc)

            lv.insert(END,f'Review result = {acc[0][0]}')
            lv.insert(END,f'Review result .csv file made at Path = {path.dirname(path.realpath(__file__))}')
            lv.insert(END,f'Name = reviewResult.csv')
            # print(f'akno my3rfosh we by3ml 3leh test => {testedResult}')

    else:
        if e.get() != '':
            review = e.get()
            review = str(review).lower()
            stpwords = {'no', 'please', 'thank', 'apologize', 'bad', 'clean', 'comfortable', 'dirty', 'enjoyed',
                        'friendly', 'glad', 'good', 'great', 'happy', 'hot', 'issues', 'nice', 'noise', 'old', 'poor',
                        'right', 'small', 'smell', 'sorry', 'wonderful'}
            restSentence = [word for word in re.split("\W+", review) if word.lower() in stpwords]
            # print(restSentence)
            # parameters begin
            no = 0
            please = 0
            thank = 0
            apologize = 0
            bad = 0
            clean = 0
            comfortable = 0
            dirty = 0
            enjoyed = 0
            friendly = 0
            glad = 0
            good = 0
            great = 0
            happy = 0
            hot = 0
            issues = 0
            nice = 0
            noise = 0
            old = 0
            poor = 0
            right = 0
            small = 0
            smell = 0
            sorry = 0
            wonderful = 0
            reviewsText = ''
            rating = ''
            # parameters end
            m = 0

            for w in restSentence:
                if w == "no":
                    no = 1
                elif w == "please":
                    please = 1
                elif w == "thank":
                    thank = 1
                elif w == "apologize":
                    apologize = 1
                elif w == "bad":
                    bad = 1
                elif w == "clean":
                    clean = 1
                elif w == "dirty":
                    dirty = 1
                elif w == "comfortable":
                    comfortable = 1
                elif w == "enjoyed":
                    enjoyed = 1
                elif w == "friendly":
                    friendly = 1
                elif w == "glad":
                    glad = 1
                elif w == "good":
                    good = 1
                elif w == "great":
                    great = 1
                elif w == "happy":
                    happy = 1
                elif w == "hot":
                    hot = 1
                elif w == "issues":
                    issues = 1
                elif w == "nice":
                    nice = 1
                elif w == "noise":
                    noise = 1
                elif w == "old":
                    old = 1
                elif w == "poor":
                    poor = 1
                elif w == "right":
                    right = 1
                elif w == "small":
                    small = 1
                elif w == "smell":
                    smell = 1
                elif w == "sorry":
                    sorry = 1
                elif w == "wonderful":
                    wonderful = 1

        revRows = []
        revRows.append(
            [no, please, thank, apologize, bad, clean, comfortable, dirty, enjoyed, friendly, glad, good, great,
             happy, hot, issues, nice, noise, old, poor, right, small, smell, sorry,
             wonderful])  # enter review here elmfrod b3d ma yt3mlo sampling
        df = pd.DataFrame(revRows, columns=cols.drop('rating'))
        df.to_csv('reviews.csv', index=False)
        decisionTreeClassifier('review_test')


scrollbar = Scrollbar(orient="horizontal")
e = Entry(root, width=100, background='white', borderwidth=1, relief=SUNKEN,xscrollcommand=scrollbar.set)
scrollbar.pack(fill="x")
scrollbar.config(command=e.xview)
e.config()
rules = Label(root,
              text='Rules for usage:\n1-You must choose Training File before doing anything\n2-If you want to Evaluate a single Text review write/paste it in the text field then click Evaluate Review\n3-If you want accuracy choose Dev File then click Evaluate Accuracy\n4-If you want to test unkown dataset choose Test File then click Show Test Results')
buttonTrain = Button(root, text="Choose Train File", command=openTrain, width=100, pady=20)
buttonDev = Button(root, text="Choose Dev File", command=openDev, width=100, pady=20)
buttonTest = Button(root, text="Choose Test File", command=openTest, width=100, pady=20)
buttonEvaluateAccuarcy = Button(root, text="Evaluate Accuracy", command=lambda: decisionTreeClassifier('dev'),
                                width=100, pady=20)
buttonEvaluateText = Button(root, text="Evaluate Review", command=lambda: decisionTreeClassifier('rev'), width=100,
                            pady=20)
buttonTestNewData = Button(root, text="Show Test Results", command=lambda: decisionTreeClassifier('test'), width=100,
                           pady=20)
clrLog = Button(root, text="Clear Log", command=clearLog, width=100,
                           pady=20)
log = Label(root,text="Log will be show Below this text:")

0
texField = Label(root,text="Text Field to write review in it:")
texField.pack()
e.pack()
rules.pack()
buttonEvaluateText.pack()
buttonTrain.pack()
buttonDev.pack()
buttonTest.pack()
buttonEvaluateAccuarcy.pack()
buttonTestNewData.pack()
log.pack()
lv.pack()
clrLog.pack()

root.mainloop()
