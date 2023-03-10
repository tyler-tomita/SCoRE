class BaseCNN(nn.Module):
    def __init__(self, num_classes, fc_hidden_size, batchnorm=True):
        super(BaseCNN, self).__init__()

        self.head_in_size = 256 * 2 * 2
        self.num_classes = num_classes
        self.fc_hidden_size = fc_hidden_size

        if batchnorm:
            self.base = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                # output size = (32, 32, 16)
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                # output size = (16, 16, 32)
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                # output size = (8, 8, 64)
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                # output size = (4, 4, 128)
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                # output size = (2, 2, 256)
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Flatten()
            )
            self.fc = nn.Sequential(
                nn.Linear(self.head_in_size, self.fc_hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(self.fc_hidden_size),
                nn.Linear(self.fc_hidden_size, self.fc_hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(self.fc_hidden_size)
            )
            self.head = nn.Linear(self.fc_hidden_size, self.num_classes)
        else:
            self.base = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                # output size = (32, 32, 16)
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                # output size = (16, 16, 32)
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                # output size = (8, 8, 64)
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                # output size = (4, 4, 128)
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                # output size = (2, 2, 256)
                nn.ReLU(),
                nn.Flatten()
            )
            self.fc = nn.Sequential(
                nn.Linear(self.head_in_size, self.fc_hidden_size),
                nn.ReLU(),
                nn.Linear(self.fc_hidden_size, self.fc_hidden_size),
                nn.ReLU()
            )
            self.head = nn.Linear(self.fc_hidden_size, self.num_classes)

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        outputs = self.base(x)
        outputs = self.fc(outputs)
        outputs = self.head(outputs)
        outputs = self.softmax(outputs)
        return outputs


class SCORE(nn.Module):
    def __init__(self, expert_class, ensembler_class):
        super(SCORE, self).__init__()

        # start with zero experts
        self.num_experts = 0
        self.experts = nn.ModuleList([])
        self.frozen = []
        self.expert_out_size = []
        self.expert_class = expert_class # default expert constructor


        # ensembler layer
        self.num_ensemblers = 0
        self.ensemblers = nn.ModuleList([])
        self.ensembler_out_size = []
        self.ensembler_class = ensembler_class # default ensembler constructor
        self.task2ensembler = [] # maps the task id to the index of its corresponding ensembler

    def forward(self, input_tensor, selected_ensemblers=range(self.num_ensemblers)):
        expert_outputs = torch.zeros((input_tensor.size(0), 1, self.expert_start_indices[-1])).to(device)
        for i, expert in enumerate(self.experts):
            expert_outputs[:, 0, self.expert_start_indices[i]:self.expert_start_indices[i+1]] = expert(input_tensor).squeeze()

        ensembler_outputs = torch.zeros((input_tensor.size(0), self.ensembler_start_indices[-1])).to(device)
        for i in selected_ensemblers:
            ensembler = self.ensemblers[i]
            ensembler_outputs[:, self.ensembler_start_indices[i]:self.ensembler_start_indices[i+1]] = ensembler(expert_outputs).squeeze()

        return ensembler_outputs

    def add_ensembler(self, ensembler_network=self.ensembler_class()):
        self.ensemblers.append(ensembler_network.to(device))
        lin_layers = []
        for i, layer in enumerate(ensembler_network):
            if type(layer) == torch.nn.modules.linear.Linear:
                lin_layers.append(i)
        self.ensembler_out_size.append(ensembler_network[lin_layers[-1]].weight.data.shape[0])
        self.ensembler_start_indices = torch.cumsum(torch.tensor([0] + self.ensembler_out_size), dim=0)
        self.num_ensemblers += 1

    def add_expert(self, expert_network=self.expert_class()):
        self.experts.append(expert_network.to(device))
        self.num_experts += 1
        self.frozen.append(False)

        # get number of output units of expert
        lin_layers = []
        for i, layer in enumerate(expert_network):
          if type(layer) == torch.nn.modules.linear.Linear:
            lin_layers.append(i)
        self.expert_out_size.append(expert_network[lin_layers[-1]].weight.data.shape[0])

        self.expert_start_indices = torch.cumsum(torch.tensor([0] + self.expert_out_size), dim=0)

        ensembler_in_size = self.expert_start_indices[-1]
        # reinitizialize ensembler networks with input size increased by size of expert output
        for ensembler_idx, ensembler in enumerate(self.ensemblers):
            ensembler_params = copy.deepcopy(ensembler.state_dict())
            ensembler_weights = copy.deepcopy(ensembler[0].weight.data)
            ensembler_bias = copy.deepcopy(ensembler[0].bias.data)
            old_input_size = ensembler_weights.shape[1]
            ensembler[0] = nn.Linear(ensembler_in_size, ensembler_weights.shape[0], bias=True).to(device)
            ensembler[0].weight.data[:, :old_input_size] = ensembler_weights
            ensembler[0].weight.data[:, old_input_size:] = 0.
            ensembler[0].bias.data = ensembler_bias

    def freeze_expert(self, expert_list):
        for i in expert_list:
            for param in self.experts[i].parameters():
                param.requires_grad = False
            self.frozen[i] = True

    def freeze_ensembler(self, ensembler_list):
        for i in ensembler_list:
            for param in self.ensemblers[i].parameters():
                param.requires_grad = False

    def unfreeze_expert(self, expert_list):
        for i in expert_list:
            for param in self.experts[i].parameters():
                param.requires_grad = True
            self.frozen[i] = False

    def unfreeze_ensembler(self, ensembler_list):
        for i in ensembler_list:
            for param in self.ensemblers[i].parameters():
                param.requires_grad = True


class NEM(nn.Module):
    def __init__(self, input_size, encoder_class, decoder_class, max_memories):
        super(NEM, self).__init__()

        self.input_size = input_size

        self.n_tasks = 0
        self.max_memories = max_memories # max memories per task
        self.n_memories = {}

        self.memory_encoder = encoder_class()
        self.memory_decoder = decoder_class()

        # initialize memory buffer
        self.memories = {}

        # self.memory_encoder = nn.Linear(self.input_size, self.encoder_size, bias=False)
        # self.memory_decoder = nn.Linear(self.encoder_size, self.input_size, bias=False)

        # self.memory_encoder = nn.Sequential(
        #     nn.Linear(self.input_size, self.encoder_size, bias=True),
        #     nn.Tanh(),
        #     nn.Linear(self.encoder_size, self.encoder_size, bias=True)
        # )
        # self.memory_decoder = nn.Sequential(
        #     nn.Linear(self.encoder_size, self.encoder_size, bias=True),
        #     nn.Tanh(),
        #     nn.Linear(self.encoder_size, self.input_size, bias=True)
        # )

    def forward(self, input_tensor):
        output_tensor = self.memory_encoder(input_tensor)
        return output_tensor

    def add_task(self):
        if self.n_tasks == 0:
            self.memories['input'] = torch.zeros((self.max_memories, self.input_size)).to(device)
            self.memories['compressed'] = torch.zeros((self.max_memories, self.encoder_size)).to(device)
            self.memories['target'] = torch.zeros((self.max_memories, ), dtype=torch.long).to(device)
            self.memories['task'] = torch.tensor([self.n_tasks for i in range(self.max_memories)], dtype=torch.short).to(device)
        else:
            self.memories['input'] = torch.cat((self.memories['input'], torch.zeros((self.max_memories, self.input_size), device=device)), dim=0)
            self.memories['compressed'] = torch.cat((self.memories['compressed'], torch.zeros((self.max_memories, self.encoder_size), device=device)), dim=0)
            self.memories['target'] = torch.cat((self.memories['target'], torch.zeros((self.max_memories, ), dtype=torch.long, device=device)))
            self.memories['task'] = torch.cat((self.memories['task'], torch.tensor([self.n_tasks for i in range(self.max_memories)], dtype=torch.short, device=device)))
        self.n_memories[self.n_tasks] = 0
        self.n_tasks += 1

    def add_memory(self, input_tensor, target_tensor):
        if self.n_memories[self.n_tasks-1] < self.max_memories:
            mem_idx = int((self.n_memories[self.n_tasks-1] % self.max_memories) + (self.n_tasks-1)*self.max_memories)
            self.memories['input'][mem_idx, :] = input_tensor
            self.memories['target'][mem_idx] = target_tensor
            self.n_memories[self.n_tasks-1] += 1

    def add_compressed(self, indices, new_memories):
        self.memories['compressed'][indices] = new_memories

    def freeze_encoder(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.parameters():
            param.requires_grad = True
