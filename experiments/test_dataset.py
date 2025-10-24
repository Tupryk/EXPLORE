from explore.datasets.dataset import ExploreDataset


dataset = ExploreDataset("data/fingerRamp_full", horizon=4, history=2,
                         state_sigma=0.1, goal_condition=True, flatten_data=False, verbose=1)

dataset_len = len(dataset)
state, action = dataset[0]

for i in range(dataset_len):
    dataset[i]
    
print("Dataset size: ", dataset_len)
print("State: ", state)
print("Action: ", action)
