from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada import make_da_pipeline
    from benchmark_utils.base_solver import DASolver
    from skada.deep import DeepCoral
    from skada.deep.base import DomainAwareModule
    import torch
    from torch import nn
    import torchvision.models as models
    from torch.optim import Adam
    import collections
    import psutil
    import GPUtil
    from skorch.callbacks import Callback


class ResNet50WithMLP(nn.Module):
    def __init__(self, n_classes=2, hidden_size=256):
        super().__init__()
     
        # Load pre-trained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # To flatten the tensor
        self.flatten = nn.Flatten()
        # Add 2-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(2048, hidden_size),
            nn.ReLU(),
        )
        self.last_layers = nn.Sequential(
            nn.Linear(hidden_size, n_classes),
            nn.Softmax(dim=1),
        )
       

    def forward(self, x, sample_weight=None):
        x = x.reshape((x.shape[0], 3, 224, 224))
        x = self.features(x)
        x = self.flatten(x)
        x = self.mlp(x)
        x = self.last_layers(x)

        return x


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):
    # Name to select the solver in the CLI and to display the results.
    name = 'DeepCORAL'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    default_param_grid = {
        'max_epochs': [20],
        'optimizer__weight_decay': [1e-5, 1e-4, 1e-3],
        'lr': [1e-2, 1e-3, 1e-4],
    }


    def get_estimator(self, n_classes, device):
        model = ResNet50WithMLP(n_classes=n_classes)
        net = DeepCoral(
            model,
            optimizer=Adam,
            reg=1,
            layer_name="mlp",
            batch_size=128,
            max_epochs=1,
            train_split=None,
            device=device,
            #callbacks=[GPUUsageCallback()],
        )

        return net

class GPUUsageCallback(Callback):
    def on_epoch_end(self, net, **kwargs):
        self.check_multi_gpu_usage()

    @staticmethod
    def check_multi_gpu_usage():
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            
            if num_gpus > 0:
                print(f"\nCUDA is available. Found {num_gpus} GPU(s).")
                
                for i in range(num_gpus):
                    torch.cuda.set_device(i)
                    print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
                    
                    gpus = GPUtil.getGPUs()
                    gpu = gpus[i]
                    print(f"  Utilization: {gpu.load * 100:.2f}%")
                    print(f"  Memory usage: {gpu.memoryUsed / gpu.memoryTotal * 100:.2f}%")
                    
                    if torch.cuda.current_device() == i:
                        print("  This GPU is currently being used by PyTorch.")
                    else:
                        print("  This GPU is available but not currently used by PyTorch.")
            else:
                print("\nCUDA is available but no GPUs are accessible.")
        else:
            print("\nCUDA is not available. Using CPU.")
        
        print(f"\nCPU usage: {psutil.cpu_percent()}%")
