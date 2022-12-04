
img_array = cv2.imread('train/open_eyes/s0001_02036_0_0_1_0_0_01.png',cv2.IMREAD_GRAYSCALE)
DataDirectory = 'train/' #training data
Classes = ['closed_eyes','open_eyes'] #list of classes
for category in Classes:
    path = os.path.join(DataDirectory,category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        backtorgb = cv2.cvtColor(img_array,cv2.COLOR_GRAY2BGR)
        break
        
img_size = 224
new_array = cv2.resize(backtorgb,(img_size,img_size))
training_Data = []

def create_training_Data():
    for category in Classes:
        path = os.path.join(DataDirectory , category)
        class_num = Classes.index(category) # 0 or 1
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                backtorgb = cv2.cvtColor(img_array,cv2.COLOR_GRAY2RGB)
                new_array = cv2.resize(backtorgb ,(img_size,img_size))
                training_Data.append([new_array,class_num])
            except Exception as e:
                pass

create_training_Data()
X = []
y = []

for features,label in training_Data:
    X.append(features)
    y.append(label)
    
X=np.array(X).reshape(-1, img_size,img_size,3)
X=X/255.0 #normalizing the data
y = np.array(y)