# GNR 638 Assignment-2

In this part of the project, we have implemented a three layer MLP model in the already existing bag of sifts and have experimented on the hidden layer size and the activation function. 

### Steps to run the code

```bash
python mlp.py --classifier mlp
```
Before running the code, we should chnage the hidden layer sizes and activation function in the function below 

```python
 def __init__(self, input_size, hidden_size1=2048, hidden_size2=1024, num_classes=21):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, num_classes)
```
and use the `train_mlp` function to get 9 trained models whose paths go by 

```bash
'./store/mlp_model_{activation}_{hidden_size1}_{hidden_size2}_600.pth'
```
where activation=['ReLU','Tanh','GELU'] and (hidden_size1,hidden_size2)=(2048,1024),(1024,512),(512,256). `train_mlp` for one case (ReLU, 2048, 1024) is done on the uploaded code. Once the 9 trained models are collected, cross_validation using K-fold method `k=5` and test accuracy of the best model can be obtained by running the `mlp.py` code.