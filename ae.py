from util import *

	    
class AE(nn.Module):
	def __init__(self, input_dim):
		super(AE, self).__init__()
		self.E = nn.Sequential(
			nn.Linear(input_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 32),
			nn.Conv1d(1, 1, (1, 3)),
			nn.ReLU(),
			nn.Conv1d(1, 1, (1, 3)),
			nn.ReLU(),
			nn.Conv1d(1, 1, (1, 3)),
			nn.ReLU(),
			nn.Conv1d(1, 1, (1, 11))
		)
		self.D = nn.Sequential(
			nn.ConvTranspose1d(1, 1, (1, 11)),
			nn.ReLU(),
			nn.ConvTranspose1d(1, 1, (1, 3)),
			nn.ReLU(),
			nn.ConvTranspose1d(1, 1, (1, 3)),
			nn.ReLU(),
			nn.ConvTranspose1d(1, 1, (1, 3)),
			nn.Linear(32, 256),
			nn.ReLU(),
			nn.Linear(256, input_dim)
		)

	def forward(self, x):
		z = self.E(x)
		output = self.D(z)
		return z, output

epochs = 10
outputs = []
losses = []

if __name__ == "__main__":
    cur_name = "input/GF/fingerprint0.txt"
    observation_sign = "u"
    observation_ids = []
    pruned_observation_ids = []
    mac_all = []
    observation_ids, mac_all = load_file_comparison(cur_name, observation_ids, observation_sign, mac_all, file_type="db", prune=False)
    pruned_observation_ids, mac_all = load_file_comparison(cur_name, pruned_observation_ids, observation_sign, mac_all, file_type="db", prune=True)
    
    x_all = np.zeros((len(pruned_observation_ids), len(mac_all)))
    x_all = np.zeros((len(pruned_observation_ids), len(mac_all)))
    for i, item in enumerate(observation_ids):
        for j, mac in enumerate(mac_all):
            if mac in item[3].keys():
                x_all[i][j] = item[3][mac]/120
            else:
                x_all[i][j] = -120/120
    
    model = AE(len(mac_all))
    loss_function = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        for record in x_all:
            cur_record = record.reshape((1, 1, 1, len(record)))
            reconstructed = model(torch.Tensor(cur_record))
            loss = loss_function(reconstructed[1], torch.Tensor(cur_record))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss)
            # print("Epoch: {}, Loss: {}".format(epoch, loss))
        outputs.append((epochs, torch.Tensor(cur_record), reconstructed[0], reconstructed[1]))