import copy
import json
import typing as t

class ExpConfig():
    name: str
    """Name of the experiment"""
    tensorboard_dir: str = "experiment_logs"
    """Directory to store tensorboard logs"""

    #
    # Dataset
    #
    dataset_name: str 
    """Name of the dataset"""
    dataset_root: str = "/Scratch/al183/datasets"
    """Root of the dataset"""
    fixed_class_order: t.Optional[t.List[int]]
    """Whether to use the same class order for all runs"""
    n_experiences: int
    """Number of experiences in the scenario"""
    input_shape: t.Tuple[int, int, int]
    """The dimensions of the image data"""
    is_image_data: bool
    """Whether the dataset is image data"""
    n_classes: int

    #
    # Loss
    #
    use_classifier_loss: bool
    """Whether to use cross entropy loss for classification"""
    classifier_loss_weight: float
    """Weight of the classifier loss"""
    use_reconstruction_loss: bool
    """Whether to use reconstruction loss for AE and VAE"""
    recon_loss_type: t.Literal["mse", "bce"]
    """Type of loss function to use"""
    reconstruction_loss_weight: float
    """Weight of the reconstruction loss"""
    use_vae_loss: bool
    """Whether to use VAE loss for VAE"""
    vae_loss_weight: float
    """Weight of the VAE loss"""

    # 
    # Architecture
    #
    latent_dims: int
    """Latent dimensions of the VAE/AE"""
    deep_generative_type: t.Literal["AE", "VAE"]
    """Type of deep generative model to use"""
    network_architecture: t.Literal["vanilla_cnn", "residual", "mlp"]
    """Type of network architecture to use"""
    embedding_module: t.Literal["None", "ResNet50"]
    """Optionally configure the experiment to embed the dataset"""
    network_cfg: t.Dict[str, t.Any]
    """Other network configuration options"""

    #
    # Training
    #
    total_task_epochs: int
    """Total number of training epochs in a task"""
    batch_size: int
    """Batch size"""
    learning_rate: float
    """Learning rate"""
    device: str
    """Device to use for training"""
    use_adam: bool
    """
    Configure the experiment to use the Adam optimizer. If false, the SGD 
    optimizer is used instead
    """

    #
    # PackNet
    #
    use_packnet: bool
    """Whether to use packnet"""
    prune_proportion: t.Union[float, t.List[float]]
    """Proportion of the network to prune"""
    retrain_epochs: int
    """Number of epochs post-pruning to retrain the network"""
    task_inference_strategy: t.Literal["task_oracle", "task_reconstruction_loss"]
    """Type of task inference strategy to use"""

    # Baselines
    # Experience replay
    use_experience_replay: bool
    replay_buffer: int

    # Synaptic intelligence
    use_synaptic_intelligence: bool
    si_lambda: float

    # Learning without forgetting
    use_learning_without_forgetting: bool
    lwf_alpha: float

    # generative replay
    use_generative_replay: bool

    def toJSON(self):
        return json.dumps(self.__dict__, sort_keys=True, indent=4)

    def __init__(self) -> None:
        # Default Values
        self.batch_size = 64
        self.learning_rate = 0.0001
        self.device = "cuda"
        self.classifier_loss_weight = 1.0
        self.reconstruction_loss_weight = 1.0
        self.recon_loss_type = "bce"
        self.embedding_module = "None"
        self.prune_proportion = 0.5

        self.use_experience_replay = False
        self.replay_buffer = 1_000

        self.use_synaptic_intelligence = False
        self.si_lambda = 1_000

        self.use_learning_without_forgetting = False
        self.lwf_alpha = 32

        self.use_generative_replay = False
        self.use_adam = True
        self.fixed_class_order = None
        self.network_cfg = {}

    # 
    # Networks
    # 

    def use_vanilla_cnn(self: 'ExpConfig') -> 'ExpConfig':
        """Configure the experiment to use a vanilla CNN"""
        self.network_architecture = "vanilla_cnn"
        self.network_cfg["base_channels"] = 128
        return self

    def use_mlp_network(self: 'ExpConfig') -> 'ExpConfig':
        """Configure the experiment to use a rectangular network"""
        self.network_architecture = "mlp"
        self.network_cfg["width"] = 512
        return self

    def use_resnet_cnn(self: 'ExpConfig') -> 'ExpConfig':
        """Configure the experiment to use a ResNet CNN"""
        self.network_architecture = "residual"
        self.latent_dims = 64
        return self

    # 
    # Scenarios
    # 

    def use_fmnist(self: 'ExpConfig') -> 'ExpConfig':
        """Configure the experiment for the Fashion-MNIST dataset"""
        self.dataset_name = "FMNIST"
        self.input_shape = (1, 32, 32)
        self.is_image_data = True
        self.n_experiences = 5
        self.n_classes = 10

        self.total_task_epochs = 20
        self.retrain_epochs = 5

        self.use_vanilla_cnn()
        self.latent_dims = 64
        return self

    def use_cifar10(self: 'ExpConfig') -> 'ExpConfig':
        """Configure the experiment for the CIFAR10 dataset"""
        self.dataset_name = "CIFAR10"
        self.input_shape = (3, 32, 32)
        self.is_image_data = True
        self.n_experiences = 5
        self.n_classes = 10

        self.total_task_epochs = 50
        self.retrain_epochs = 10

        self.use_resnet_cnn()
        self.latent_dims = 128
        return self

    def use_cifar100(self: 'ExpConfig') -> 'ExpConfig':
        """Configure the experiment for the CIFAR100 dataset"""
        self.dataset_name = "CIFAR100"
        self.input_shape = (3, 32, 32)
        self.is_image_data = True
        self.n_experiences = 10
        self.n_classes = 100

        self.total_task_epochs = 100
        self.retrain_epochs = 30

        self.use_resnet_cnn()
        self.latent_dims = 256
        return self

    def use_core50(self: "ExpConfig") -> 'ExpConfig':
        """Configure the experiment for the Core50 dataset"""
        self.dataset_name = "CORe50_NC"
        self.input_shape = (3, 32, 32)
        self.is_image_data = True
        self.n_experiences = 10
        self.n_classes = 50

        self.total_task_epochs = 5
        self.retrain_epochs = 2
        
        self.use_resnet_cnn()
        self.latent_dims = 512

        return self

    def use_embedded_cifar100(self: 'ExpConfig') -> 'ExpConfig':
        """Configure the experiment to use the embedded CIFAR100 dataset"""
        self.dataset_name = "CIFAR100"

        self.prune_proportion = 0.5

        self.embedding_module = "ResNet50"
        self.input_shape = (2048,)
        self.is_image_data = False
        self.recon_loss_type = "mse"

        self.n_experiences = 10
        self.n_classes = 100

        self.total_task_epochs = 100
        self.retrain_epochs = 30

        self.use_mlp_network()
        self.latent_dims = 256

        return self

    def use_embedded_core50(self: 'ExpConfig') -> 'ExpConfig':
        """Configure the experiment to use the embedded CIFAR100 dataset"""
        self.dataset_name = "CORe50_NC"

        self.prune_proportion = 0.5

        self.embedding_module = "ResNet50"
        self.input_shape = (2048,)
        self.is_image_data = False
        self.recon_loss_type = "mse"

        self.n_experiences = 10
        self.n_classes = 50

        self.total_task_epochs = 10
        self.retrain_epochs = 3
        self.use_mlp_network()
        self.latent_dims = 512

        return self


    def use_auto_encoder(self: 'ExpConfig') -> 'ExpConfig':
        """Configure the experiment to use an AutoEncoder"""
        self.deep_generative_type = "AE"
        self.use_classifier_loss = True
        self.use_reconstruction_loss = True
        self.use_vae_loss = False
        return self

    def use_variational_auto_encoder(self: 'ExpConfig') -> 'ExpConfig':
        """Configure the experiment to use a variational AutoEncoder"""
        self.deep_generative_type = "VAE"
        self.use_classifier_loss = True
        self.use_reconstruction_loss = True
        self.use_vae_loss = True
        self.vae_loss_weight = 0.001
        return self

    def enable_packnet(self: 'ExpConfig') -> 'ExpConfig':
        """Configure the experiment to use PackNet"""
        self.use_packnet = True
        self.task_inference_strategy = "task_oracle"
        return self

    def use_cumulative_learning(self: 'ExpConfig') -> 'ExpConfig':
        """Configure the experiment to not do any continual learning"""
        self.use_packnet = False
        self.n_experiences = 1
        return self

    def copy(self: 'ExpConfig') -> 'ExpConfig':
        """Create a copy of the experiment configuration"""
        return copy.deepcopy(self)
