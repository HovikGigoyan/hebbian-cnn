Train the model
python train.py --config configs/base.yaml

This will load parameters from the YAML file, 
initialize the model,
start training,

Evaluate the model
python evaluate.py --config configs/base.yaml
This will load the trained model,
run evaluation on the dataset,
output performance metrics