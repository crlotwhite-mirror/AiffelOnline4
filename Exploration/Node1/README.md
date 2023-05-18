## **Code Peer Review Templete**
------------------
- 코더 : 김동규
- 리뷰어 : 김설아

## **PRT(PeerReviewTemplate)**
------------------
-[O] 1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?  
-[O] 2. 주석을 보고 작성자의 코드가 이해되었나요?

```
# Task 3: change df_y to np.array

# I tried to check the type of "df_y"
# But it is already np.ndarray. Again...
print(type(df_y)) 

# So I just made a variable.
y = df_y
```
```
# Task 4: split train data and test data
# To split datasets, I need to use the function called "train_test_split"
from sklearn.model_selection import train_test_split

# To clean the code, I use bracket, but it is no problem for the code.
# It is one of the code convention for some companies. :)
X_train, X_test, y_train, y_test = (
    train_test_split(X, y, test_size=0.2, random_state=42)
)
# About test_size=0.2 and random_state, I just follow sample code.
# Because I mostly used that proportion.


# Okay, let's check the result.
X_train[0:4], X_test[0:4], y_train[0:4], y_test[0:4]
```
와 같이 확인과정, 변수 선정 이유 등을 자세하게 설명해주셨습니다.

-[X] 3. 코드가 에러를 유발할 가능성이 있나요?
```
print(type(df_X)) 

# So I just made a variable.
X = df_X
```
타입을 확인하고 진행함.
```
# Task 5: prepare the model

# I define the simple line function.
# Because X was a matrix, so I tried to use 'dot()'.
# W.shape will be (1, 10)
def model(X, W, b):
    y = np.dot(W, X.T) + b
    return y
```
```

def MSE(a, b):
    return ((a - b) ** 2).mean() # Using mean(), I can remove

# And I imported the MSE function in the loss
def loss(X, W, b, y):
    pred = model(X, W, b) # get prediction
    L = MSE(pred, y) # get loss
    return L
```
```
# Task 7: implementation the gradient function

# To implementation the gradient descent, 
#I used Definition of Differential Coefficient

def gradient(X, W, b, y):
    N = len(y) # get length
    
    y_pred = model(X, W, b) # get predictions
    dW = 1/N * 2 * X.T.dot(y_pred - y) # get W grad
    db = (loss(X, W, b + 0.0001, y) - loss(X, W, b, y)) / 0.0001 # get b grad
    return dW, db
```
함수의 선언 순서를 지킴.


-[O] 4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?

```
# Task 9: train the model

# Initialize the model
# I choose random initialization.
W = np.random.rand(10) 
b = np.random.rand(1)

losses = []

# Train the model: Take 1
for i in range(1, 20001):
    dW, db = gradient(X_train, W, b, y_train)   # Predict y and return gradient
        
    W -= lr * dW         # Update W
    b -= lr * db         # Update b
    L = loss(X_train, W, b, y_train) # Get loss
    
    # early stopper
    if i != 1 and losses[-1] == L: 
        print('==========')
        print(f'Stoped at {i}')
        print('Iteration %d : Loss %0.4f' % (i, L))
        break
        
    losses.append(L)     # Record loss
    
    if i % 1000 == 0:
        print('Iteration %d : Loss %0.4f' % (i, L))

```
loss 값에 감소가 없을 때, 의미없는 작업을 하지 않기 위한 코드에 대해
리뷰어의 이해를 돕는 간결하고 확실한 설명을 해주셨습니다.

-[O] 5. 코드가 간결한가요?
```
# I use train_test_split function for divide data
X_train, X_test, y_train, y_test = (
    train_test_split(X, y, test_size=0.2, random_state=42)
)
```
괄호를 사용하여 가독성을 높였습니다.


## **참고링크 및 코드 개선 여부**
------------------
- ChatGPT에 'Please recommend random state and test size that are mainly used in linear regression models'로 질문해 
코더님이 사용하신 test_size와 random_state가 보편적으로 사용되는 것을 확인하였습니다
```
X_train, X_test, y_train, y_test = (
    train_test_split(X, y, test_size=0.2, random_state=42)
)
```
    
