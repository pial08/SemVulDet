from sklearn.model_selection import train_test_split
import json
import random

with open("../dataset/moco-pretrain/non-vulnerables.json") as f:
    #lines = f.readlines()
            
    for line in f:
        js=json.loads(line.strip())
        #print(js)
        print("------------------------------------------------")

#print(js[7899]["code"])

benignList = []
for b in js:
    benignList.append({"code": b["code"], "target": 0})


#print(benignList[15000])
with open("../dataset/moco-pretrain/vulnerables.json") as f:
#lines = f.readlines()
        
    for line in f:
        js=json.loads(line.strip())
        #print(js)
        print("------------------------------------------------")
vulnerableList = []
for v in js:
    vulnerableList.append({"code": v["code"], "target": 1})
    
print(vulnerableList[0]['code'])

finalList = benignList + vulnerableList


random.shuffle(finalList)
print("Len of bernigh code ", len(benignList))
print("Len of vulnerable list ", len(vulnerableList))




train, test = train_test_split(finalList, test_size=0.3)
val, test = train_test_split(test, test_size=0.5)



print(len(train))
#train = json.dumps(train, indent=0)
#test = json.dumps(test, indent=0)
#val = json.dumps(val, indent=0)

#print(train)

print(len(train))

with open("training.jsonl", "a") as out_file:
    for obj in train:    
        #print(obj)
        json.dump(obj, out_file)
        out_file.write('\n')

with open("testing.jsonl", "a") as out_file:
    for obj in test:    
        #print(obj)
        json.dump(obj, out_file)
        out_file.write('\n')


with open("validating.jsonl", "a") as out_file:
    for obj in val:    
        #print(obj)
        json.dump(obj, out_file)
        out_file.write('\n')


#jsonFile = open("training.jsonl", "w")
#jsonFile.write(train)




#print(final)


