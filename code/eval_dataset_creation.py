from sklearn.model_selection import train_test_split
import json
import random
"""
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

"""


#print(benignList[15000])
with open("evaluation_data.txt") as f:
#lines = f.readlines()
        
    for line in f:
        #js=json.loads(line.strip())
        js = f.read()
        #print(js)
        print("------------------------------------------------")

print("Total number of functions in Evaluation Dataset ... ",  len(js.split("#########################################")))

vulnerableList = []
vul_counter = 0
ben_counter = 0
js = js.split("#########################################")
for v in js:
    v = v.split("@")[0]
    print("Printing v", v)
    vulnerableList.append({"code": v, "target": 1})
    print("Appended ...")
          
    
print(vulnerableList[0]['code'], vulnerableList[0]['target'])


finalList = vulnerableList


#random.shuffle(finalList)
#print("Len of benign code ", ben_counter)
#print("Len of vulnerable list ", vul_counter)




#train, test = train_test_split(finalList, test_size=0.2)
#val, test = train_test_split(test, test_size=0.5)



#print("Total datasets ", vul_counter + ben_counter)
#train = json.dumps(train, indent=0)
#test = json.dumps(test, indent=0)
#val = json.dumps(val, indent=0)

#print(train)

#print(len(train))

with open("evaluation.jsonl", "a") as out_file:
    for obj in finalList:    
        #print(obj)
        json.dump(obj, out_file)
        out_file.write('\n')


#jsonFile = open("training.jsonl", "w")
#jsonFile.write(train)




#print(final)


