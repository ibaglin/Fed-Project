# Fed-Project

The purpose of this project is to determine Federated Learnings robustness against poor communication, poor data quality and lack of training data. Also to look at protection methods for Model Poisoning and Deep Leakage. To protect the model from Model Poisoning an accuracy check will detect adversarial nodes before aggregation. Synthetic data will be used to train the model in the 'Fed_Deep Leakage.py' testbed to stop private training data from being recovered.

--- Fed_testbed.py ---

- A Federated Learning Testbed trained using the CIFAR-10 dataset. Accuracy is printed at the end of each round.
- Change variable 'rounds' and 'num_clients' to change the number of training rounds and nodes. (recommended 'rounds' = 30 and 'num_clients' = 5 )
- To reduce the data quality change the variable 'poor_data' to 1 and the amount of Gaussian blur applied with the variable 'std' (recommended 0 <-> 10)
- To introduce communication noise change the variable state to 1 and change the value of 'std' (recommended 0 <-> 0.2)
- To reduce the amount of data used to train the model increase 'scale_reduction' 
- Change 'add_num' to the number of adversarial nodes needed for Model Poisoning. (Model Poisoning protection included.)

--- Non_Fed.py ---

- A Non-Federated Learning Testbed trained using the CIFAR-10 dataset. Accuracy is printed at the end of each round.
- Change variable 'rounds' and 'num_clients' to change the number of training rounds and nodes. (recommended 'rounds' = 30 and 'num_clients' = 5 )
- To reduce the data quality change the variable 'poor_data' to 1 and the amount of Gaussian blur applied with the variable 'std' (recommended 0 <-> 10)
- To introduce communication noise change the variable state to 1 and change the value of 'std' (recommended 0 <-> 0.2)
- To reduce the amount of data used to train the model decrease 'imagelabel' 

--- Deep Leakage Experiment ---

- 'images' folder contains 400 synthetic MNIST training images
- 'images2' folder contains 100 synthetic MNIST test images
- 'test_file.csv' and 'dataset_file.csv' contains matching labels for the synthetic training and test data
- 'Generator_epoch_200.pth' is a pre-trained MNIST generator.
- 'model_scripted.pt' is a saved pre-trained model using the real data.
- 'label.py' prints predicted labels for all the synthetic data in 'dataset_file.csv' ( The labels in the csv file should initially be zero)
- 'print_result.py' generates 400 synthetic images and saves them to the current directory.
- 'Fed_Deep_Leakage.py' is a Federated testbed which trains the model using the synthetic MNIST data.

