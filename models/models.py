import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils import init_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SCORE(nn.Module):
    def __init__(self, expert_class, ensembler_class, input_size, hidden_size_expert, output_size_expert, hidden_size_ensembler, output_size_ensembler):
        super(SCORE, self).__init__()

        self.input_size = input_size
        self.hidden_size_expert = hidden_size_expert
        self.output_size_expert = output_size_expert
        self.hidden_size_ensembler = hidden_size_ensembler
        self.output_size_ensembler = output_size_ensembler

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

    def forward(self, input_tensor, selected_ensemblers=None):
        if selected_ensemblers is None:
            selected_ensemblers = range(self.num_ensemblers)

        expert_outputs = torch.zeros((input_tensor.size(0), 1, self.expert_start_indices[-1])).to(device)
        for i, expert in enumerate(self.experts):
            expert_outputs[:, 0, self.expert_start_indices[i]:self.expert_start_indices[i+1]] = expert(input_tensor).squeeze()

        ensembler_outputs = torch.zeros((input_tensor.size(0), self.ensembler_start_indices[-1])).to(device)
        for i in selected_ensemblers:
            ensembler = self.ensemblers[i]
            ensembler_outputs[:, self.ensembler_start_indices[i]:self.ensembler_start_indices[i+1]] = ensembler(expert_outputs).squeeze()

        return ensembler_outputs, expert_outputs

    def add_ensembler(self, ensembler_network=None):
        if ensembler_network is None:
            ensembler_network = self.ensembler_class(self.expert_start_indices[-1], self.hidden_size_ensembler, self.output_size_ensembler).net

        self.ensemblers.append(ensembler_network.to(device))
        lin_layers = []
        for i, layer in enumerate(ensembler_network):
            if type(layer) == torch.nn.modules.linear.Linear:
                lin_layers.append(i)
        self.ensembler_out_size.append(ensembler_network[lin_layers[-1]].weight.data.shape[0])
        self.ensembler_start_indices = torch.cumsum(torch.tensor([0] + self.ensembler_out_size), dim=0)
        self.num_ensemblers += 1

    def remove_ensembler(self):
        self.ensemblers = self.ensemblers[:-1]
        self.ensembler_out_size = self.ensembler_out_size[:-1]
        self.ensembler_start_indices = torch.cumsum(torch.tensor([0] + self.ensembler_out_size), dim=0)
        self.num_ensemblers -= 1

    def add_expert(self, expert_network=None):
        if expert_network is None:
            expert_network = self.expert_class(self.input_size, self.hidden_size_expert, self.output_size_expert).net

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
            ensembler_weights = copy.deepcopy(ensembler[0].weight.data)
            ensembler_bias = copy.deepcopy(ensembler[0].bias.data)
            old_input_size = ensembler_weights.shape[1]
            ensembler[0] = nn.Linear(ensembler_in_size, ensembler_weights.shape[0], bias=True).to(device)
            if ensembler_idx != self.task2ensembler[-1]:
                ensembler[0].weight.data[:, :old_input_size] = ensembler_weights
                ensembler[0].weight.data[:, old_input_size:] = 0.
                ensembler[0].bias.data = ensembler_bias
            else:
                ensembler.apply(init_weights)
                # ensembler[0].weight.data[:, :old_input_size] = 0.
                # ensembler[0].bias.data = ensembler_bias
                

    def freeze_expert(self, expert_list):
        for i in expert_list:
            for param in self.experts[i].parameters():
                param.requires_grad = False
            self.experts[i].eval()
            self.frozen[i] = True

    def freeze_ensembler(self, ensembler_list):
        for i in ensembler_list:
            for param in self.ensemblers[i].parameters():
                param.requires_grad = False
            self.ensemblers[i].eval()

    def unfreeze_expert(self, expert_list):
        for i in expert_list:
            for param in self.experts[i].parameters():
                param.requires_grad = True
            self.experts[i].train()
            self.frozen[i] = False

    def unfreeze_ensembler(self, ensembler_list):
        for i in ensembler_list:
            for param in self.ensemblers[i].parameters():
                param.requires_grad = True
            self.ensemblers[i].train()


class SCORE2(nn.Module):
    def __init__(self, expert_class, ensembler_class):
        super(SCORE2, self).__init__()


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

    def forward(self, input_tensor, selected_ensemblers=None):
        if selected_ensemblers is None:
            selected_ensemblers = range(self.num_ensemblers)

        expert_outputs = torch.zeros((input_tensor.size(0), 1, self.expert_start_indices[-1])).to(device)
        for i, expert in enumerate(self.experts):
            expert_outputs[:, 0, self.expert_start_indices[i]:self.expert_start_indices[i+1]] = expert(input_tensor).squeeze()

        ensembler_outputs = torch.zeros((input_tensor.size(0), self.ensembler_start_indices[-1])).to(device)
        for i in selected_ensemblers:
            ensembler = self.ensemblers[i]
            ensembler_outputs[:, self.ensembler_start_indices[i]:self.ensembler_start_indices[i+1]] = ensembler(expert_outputs).squeeze()

        return ensembler_outputs, expert_outputs

    def add_ensembler(self, ensembler_network=None):
        # ensembler class must accept an input size, since the input size varies as more experts are added
        if ensembler_network is None:
            ensembler_network = self.ensembler_class(self.expert_start_indices[-1]).net

        self.ensemblers.append(ensembler_network.to(device))
        lin_layers = []
        for i, layer in enumerate(ensembler_network):
            if type(layer) == torch.nn.modules.linear.Linear:
                lin_layers.append(i)
        self.ensembler_out_size.append(ensembler_network[lin_layers[-1]].weight.data.shape[0])
        self.ensembler_start_indices = torch.cumsum(torch.tensor([0] + self.ensembler_out_size), dim=0)
        self.num_ensemblers += 1

    def remove_ensembler(self, idx=-1):
        del(self.ensemblers[idx])
        del(self.ensembler_out_size[idx])
        self.ensembler_start_indices = torch.cumsum(torch.tensor([0] + self.ensembler_out_size), dim=0)
        self.num_ensemblers -= 1

    def add_expert(self, expert_network=None):
        if expert_network is None:
            # expert class must have an `output_size` attribute to inform the ensembler the size of the input
            expert_network = self.expert_class()

        self.experts.append(expert_network.net.to(device))
        self.num_experts += 1
        self.frozen.append(False)

        # get number of output units of expert
        self.expert_out_size.append(expert_network.output_size)

        self.expert_start_indices = torch.cumsum(torch.tensor([0] + self.expert_out_size), dim=0)

        ensembler_in_size = self.expert_start_indices[-1]
        # reinitizialize ensembler networks with input size increased by size of expert output
        for ensembler_idx, ensembler in enumerate(self.ensemblers):
            ensembler_weights = copy.deepcopy(ensembler[0].weight.data)
            ensembler_bias = copy.deepcopy(ensembler[0].bias.data)
            old_input_size = ensembler_weights.shape[1]
            ensembler[0] = nn.Linear(ensembler_in_size, ensembler_weights.shape[0], bias=True).to(device)
            if ensembler_idx != self.task2ensembler[-1]:
                ensembler[0].weight.data[:, :old_input_size] = ensembler_weights
                ensembler[0].weight.data[:, old_input_size:] = 0.
                ensembler[0].bias.data = ensembler_bias
            else:
                ensembler[0].weight.data[:, :old_input_size] = 0.
                ensembler[0].bias.data = ensembler_bias
                

    def freeze_expert(self, expert_list):
        for i in expert_list:
            for param in self.experts[i].parameters():
                param.requires_grad = False
            self.experts[i].eval()
            self.frozen[i] = True

    def freeze_ensembler(self, ensembler_list):
        for i in ensembler_list:
            for param in self.ensemblers[i].parameters():
                param.requires_grad = False
            self.ensemblers[i].eval()

    def unfreeze_expert(self, expert_list):
        for i in expert_list:
            for param in self.experts[i].parameters():
                param.requires_grad = True
            self.experts[i].train()
            self.frozen[i] = False

    def unfreeze_ensembler(self, ensembler_list):
        for i in ensembler_list:
            for param in self.ensemblers[i].parameters():
                param.requires_grad = True
            self.ensemblers[i].train()


class NEM(nn.Module):
    # Neural Episodic Memory model stores past training examples and learns to
    # discriminate examples from different tasks
    def __init__(self, input_size, encoder_class, decoder_class, max_memories):
        super(NEM, self).__init__()

        self.input_size = input_size

        self.n_tasks = 0
        self.max_memories = max_memories # max memories per task
        self.n_memories = {}

        self.memory_encoder = encoder_class()
        self.memory_decoder = decoder_class()

        self.latent_size = self.memory_encoder.net[-1].weight.data.shape[0]

        # initialize memory buffer
        self.memories = {}

        # self.memory_encoder = nn.Linear(self.input_size, self.latent_size, bias=False)
        # self.memory_decoder = nn.Linear(self.latent_size, self.input_size, bias=False)

        # self.memory_encoder = nn.Sequential(
        #     nn.Linear(self.input_size, self.latent_size, bias=True),
        #     nn.Tanh(),
        #     nn.Linear(self.latent_size, self.latent_size, bias=True)
        # )
        # self.memory_decoder = nn.Sequential(
        #     nn.Linear(self.latent_size, self.latent_size, bias=True),
        #     nn.Tanh(),
        #     nn.Linear(self.latent_size, self.input_size, bias=True)
        # )

    def forward(self, input_tensor):
        output_tensor = self.memory_encoder(input_tensor)
        return output_tensor

    def add_task(self):
        if self.n_tasks == 0:
            self.memories['input'] = torch.zeros((self.max_memories, self.input_size)).to(device)
            self.memories['compressed'] = torch.zeros((self.max_memories, self.latent_size)).to(device)
            self.memories['target'] = torch.zeros((self.max_memories, ), dtype=torch.long).to(device)
            self.memories['task'] = torch.tensor([self.n_tasks for i in range(self.max_memories)], dtype=torch.short).to(device)
        else:
            self.memories['input'] = torch.cat((self.memories['input'], torch.zeros((self.max_memories, self.input_size), device=device)), dim=0)
            self.memories['compressed'] = torch.cat((self.memories['compressed'], torch.zeros((self.max_memories, self.latent_size), device=device)), dim=0)
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


class NEM2(nn.Module):
    # Neural Episodic Memory model stores past training examples and learns to
    # discriminate examples from different tasks
    def __init__(self, input_shape, encoder_class, max_memories):
        super(NEM2, self).__init__()

        self.input_shape = input_shape

        self.n_tasks = 0
        self.max_memories = max_memories # max memories per task
        self.n_memories = {}

        self.memory_encoder = encoder_class()

        # self.latent_size = self.memory_encoder.net[-1].weight.data.shape[0]

        # initialize memory buffer
        self.memories = {}

    def forward(self, input_tensor):
        output_tensor = self.memory_encoder(input_tensor)
        return output_tensor

    def add_task(self):
        if self.n_tasks == 0:
            self.memories['input'] = torch.zeros((self.max_memories, self.input_shape[0], self.input_shape[1], self.input_shape[2])).to(device)
            self.memories['target'] = torch.zeros((self.max_memories, ), dtype=torch.long).to(device)
            self.memories['task'] = torch.tensor([self.n_tasks for i in range(self.max_memories)], dtype=torch.short).to(device)
        else:
            self.memories['input'] = torch.cat((self.memories['input'], torch.zeros((self.max_memories, self.input_shape[0], self.input_shape[1], self.input_shape[2]), device=device)), dim=0)
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

    def freeze_encoder(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.parameters():
            param.requires_grad = True


class DynaMoE(nn.Module):
    def __init__(self, expert_class, input_size, hidden_size_expert, hidden_size_expert_2, output_size_expert):
        super(DynaMoE, self).__init__()

        self.input_size = input_size
        self.hidden_size_expert = hidden_size_expert
        self.hidden_size_expert_2 = hidden_size_expert_2
        self.output_size_expert = output_size_expert

        # start with zero experts
        self.num_experts = 0
        self.experts = nn.ModuleList([])
        self.frozen = []
        self.expert_out_size = []
        self.expert_class = expert_class # default expert constructor
        self.task2expert = []

    def forward(self, input_tensor):

        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_outputs.append(expert(input_tensor).squeeze())

        return expert_outputs

    def add_expert(self, expert_network=None):
        if expert_network is None:
            expert_network = self.expert_class(self.input_size, self.hidden_size_expert, self.hidden_size_expert_2, self.output_size_expert).net

        self.experts.append(expert_network.to(device))
        self.num_experts += 1
        self.frozen.append(False)

    def freeze_expert(self, expert_list):
        for i in expert_list:
            for param in self.experts[i].parameters():
                param.requires_grad = False
            self.experts[i].eval()
            self.frozen[i] = True

    def unfreeze_expert(self, expert_list):
        for i in expert_list:
            for param in self.experts[i].parameters():
                param.requires_grad = True
            self.experts[i].train()
            self.frozen[i] = False


class EWC(nn.Module):
    # elastic weight consolidation
    def __init__(self, input_size, hidden_size_expert, output_size_expert, hidden_size_head, num_classes, num_experts, num_heads):
        super(EWC, self).__init__()

        self.input_size = input_size
        self.hidden_size_expert = hidden_size_expert
        self.output_size_expert = output_size_expert
        self.hidden_size_head = hidden_size_head
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.num_heads = num_heads

        self.experts = nn.ModuleList([])
        for i in range(self.num_experts):
            self.experts.append(
                nn.Sequential(
                    nn.Linear(input_size, self.hidden_size_expert, bias=True),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size_expert, self.hidden_size_expert, bias=True),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size_expert, self.output_size_expert, bias=True),
                    nn.ReLU(),
                )
            )
        self.heads = nn.ModuleList([])
        for i in range(self.num_heads):
            # self.heads.append(nn.Linear(self.num_experts*self.output_size_expert, self.num_classes, bias=True))
            self.heads.append(
                nn.Sequential(
                    nn.Linear(self.num_experts*self.output_size_expert, self.hidden_size_head, bias=True),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size_head, self.num_classes, bias=True),
                ).to(device)
            )
        
        
        # self.ensemble = nn.Sequential(
        #     nn.Linear(self.num_experts*self.output_size_expert, self.hidden_size_ensembler, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_size_ensembler, self.num_classes, bias=True),
        #     nn.Linear(self.num_experts*self.output_size_expert, self.num_classes, bias=True)
        #     ).to(device)
        # self.ensemble = nn.Linear(self.num_experts*self.hidden_size, self.num_classes, bias=True)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.F = {} # fisher info
        self.task_params = {} # state of all network parameters after each task

    def forward(self, input_tensor, head_idx=-1):
        expert_outputs = torch.zeros((input_tensor.size(0), 1, self.num_experts*self.output_size_expert)).to(device)

        for i, expert in enumerate(self.experts):
            expert_outputs[:, 0, i*self.output_size_expert:(i+1)*self.output_size_expert] = expert(input_tensor).squeeze()

        output = self.softmax(self.heads[head_idx](expert_outputs))

        return output

    def store_task_params(self, task_id):
        self.task_params[task_id] = [param.detach().clone() for param in self.experts.parameters()]

    def fisher_information(self, task_id, inputs):
        # compute according to equation (6) in the appendix of https://arxiv.org/pdf/1904.07734.pdf
        n = inputs.shape[0]

        self.F[task_id] = [torch.zeros_like(param) for param in self.task_params[task_id]]

        for i in range(n):
            self.zero_grad()

            log_likelihoods = self.forward(inputs[i], task_id).squeeze()
            argmax = log_likelihoods.topk(1)[1]
            pred = log_likelihoods[argmax]

            pred.backward()

            for layer_ind, param in enumerate(self.experts.parameters()):
                self.F[task_id][layer_ind] += 1/n*param.grad**2     


# class GaborFeatureExtractor(nn.Module):
#     def __init__(self, img_size, batchnorm=True):
#         super(GaborFeatureExtractor, self).__init__()

#         self.fc_in_size = int(256 * img_size/16 * img_size/16)
#         if batchnorm:
#             self.base = nn.Sequential(
#                 nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
#                 # output size = (32, 32, 16)
#                 nn.ReLU(),
#                 nn.BatchNorm2d(16),
#                 nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
#                 # output size = (16, 16, 32)
#                 nn.ReLU(),
#                 nn.BatchNorm2d(32),
#                 nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#                 # output size = (8, 8, 64)
#                 nn.ReLU(),
#                 nn.BatchNorm2d(64),
#                 nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#                 # output size = (4, 4, 128)
#                 nn.ReLU(),
#                 nn.BatchNorm2d(128),
#                 nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#                 # output size = (2, 2, 256)
#                 nn.ReLU(),
#                 nn.BatchNorm2d(256),
#                 nn.Flatten()
#             )
#             self.fc = nn.Sequential(
#                 nn.Linear(self.fc_in_size, 100),
#                 nn.ReLU(),
#                 nn.BatchNorm1d(100),
#                 nn.Linear(100, 10),
#                 nn.ReLU(),
#                 nn.BatchNorm1d(10)
#             )
#             self.head = nn.Linear(10, 1)
#         else:
#             self.base = nn.Sequential(
#                 nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
#                 # output size = (32, 32, 16)
#                 nn.ReLU(),
#                 nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
#                 # output size = (16, 16, 32)
#                 nn.ReLU(),
#                 nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#                 # output size = (8, 8, 64)
#                 nn.ReLU(),
#                 nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#                 # output size = (4, 4, 128)
#                 nn.ReLU(),
#                 nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#                 # output size = (2, 2, 256)
#                 nn.ReLU(),
#                 nn.Flatten()
#             )
#             self.fc = nn.Sequential(
#                 nn.Linear(self.fc_in_size, 100),
#                 nn.ReLU(),
#                 nn.Linear(100, 10),
#                 nn.ReLU(),
#             )
#             self.head = nn.Linear(10, 1)
        
#     def forward(self, x):
#         outputs = self.base(x)
#         outputs = self.fc(outputs)
#         outputs = self.head(outputs)
#         return outputs
    
# class GaborFeatureExtractor2(nn.Module):
#     def __init__(self):
#         super(GaborFeatureExtractor2, self).__init__()

#         self.base = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), # hw/2
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # hw/4
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # hw/8
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # hw/16
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # hw/32
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # hw/64
#             nn.Flatten() # (256 * h/64 * w/64)
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(1024, 256),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Linear(256, 64),
#             nn.ReLU(),
#             nn.BatchNorm1d(64)
#         )
#         self.frequency_head = nn.Linear(64, 1)
#         self.orientation_head = nn.Linear(64, 1)
#         self.color_head = nn.Sequential(
#             nn.Linear(64, 3),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.base(x)
#         x = self.fc(x)
#         frequency_outputs = self.frequency_head(x)
#         orientation_outputs = self.orientation_head(x)
#         color_outputs = self.color_head(x)
#         return frequency_outputs, orientation_outputs, color_outputs
    
class GaborFeatureExtractor(nn.Module):
    def __init__(self):
        super(GaborFeatureExtractor, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3),
            # output size = (h/2, w/2, 16)
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            # output size = (h/4, w/4, 16)
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            # output size = (h/8, w/8, 32)
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            # output size = (h/16, w/16, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            # output size = (4, 4, 64)
            nn.Flatten()
        )
        self.frequency_head = nn.Sequential(
            nn.Linear(4 * 4 * 64, 1),
            nn.Sigmoid()
        )
        self.orientation_head = nn.Sequential(
            nn.Linear(4 * 4 * 64, 1),
            nn.Sigmoid()
        )
        self.color_head = nn.Sequential(
            nn.Linear(4 * 4 * 64, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        frequency_outputs = self.frequency_head(x)
        orientation_outputs = self.orientation_head(x)
        color_outputs = self.color_head(x)
        return frequency_outputs, orientation_outputs, color_outputs
    

class GaborFeatureExtractorAE(nn.Module):
    def __init__(self):
        super(GaborFeatureExtractorAE, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3),
            # output size = (h/2, w/2, 16)
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            # output size = (h/4, w/4, 16)
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            # output size = (h/8, w/8, 32)
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            # output size = (h/16, w/16, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            # output size = (4, 4, 64)
            nn.Flatten()
        )
        self.encoder = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4)
        
        self.decoder = nn.Sequential(
            nn.Upsample((8, 8)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.Sigmoid()
        )

        self.frequency_head = nn.Sequential(
            nn.Linear(4 * 4 * 64, 1),
            nn.Sigmoid()
        )
        self.orientation_head = nn.Sequential(
            nn.Linear(4 * 4 * 64, 1),
            nn.Sigmoid()
        )
        self.color_head = nn.Sequential(
            nn.Linear(4 * 4 * 64, 3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        frequency_outputs = self.frequency_head(x)
        orientation_outputs = self.orientation_head(x)
        color_outputs = self.color_head(x)
        decoder_outputs = x.view((batch_size, 64, 4, 4))
        decoder_outputs = self.decoder(decoder_outputs)
        return frequency_outputs, orientation_outputs, color_outputs, decoder_outputs
    
class FeatureExtractor(nn.Module):
    def __init__(self, frequency_extractor, orientation_extractor, color_extractor):
        super(FeatureExtractor, self).__init__()

        self.net = nn.ModuleList(
            [
            frequency_extractor,
            orientation_extractor,
            color_extractor
            ]
        )

    def forward(self, inputs):
        outputs = []
        for module in self.net:
            outputs.append(module(inputs))
        outputs = torch.cat(outputs, dim=1)
        return outputs
    