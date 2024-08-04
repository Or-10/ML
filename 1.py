import csv

hypo=[]
data = []

with open('enjoysport.csv') as csv_file:
    fd = csv.reader(csv_file)

    print("data set loaded, samples are: ")
    for x in fd:
        print(x)

        if(x[-1]== "Yes"):
            data.append(x)
    print("+ve examples are")
    for x1 in data:
        print(x1)

    row=len(data)
    col=len(data[0]) if data else 0
    
    hypo = data[0][:col-1]

    for i in range(1,row):
        for j in range(col-1):
            if hypo[j]!= data[i][j]:
                hypo[j] = '?'
    print("find S hypo is :")
    print(hypo)
           