from explore.datasets.dataset import ExploreDataset


dataset = ExploreDataset("data/fingerRamp_full", horizon=4, history=2, goal_condition=True, verbose=1)

dataset_len = len(dataset)

for i in range(dataset_len):
    dataset[i]
    
action, state, goal_cond = dataset[7]

print("Dataset size: ", dataset_len)
print("State: ", state)
print("Action: ", action)
