import json
import typing as t

class VanillaCNNConfig():
    base_channels: int
    """The number of channels in the base convolutional layer"""

class ExperimentConfiguration():
    name: str
    """Name of the experiment"""
    tensorboard_dir: str
    """Directory to store tensorboard logs"""

    #
    # Dataset
    #
    dataset_name: str
    """Name of the dataset"""
    dataset_root: str
    """Root of the dataset"""
    fixed_class_order: bool
    """Whether to use the same class order for all runs"""
    n_experiences: int
    """Number of experiences in the scenario"""
    input_channel_size: int
    """The size of the input channel. 3 for RGB, 1 for grayscale"""
    is_image_data: bool
    """Whether the dataset is image data"""

    #
    # Loss
    #
    use_classifier_loss: bool
    """Whether to use cross entropy loss for classification"""
    classifier_loss_weight: float
    """Weight of the classifier loss"""
    use_reconstruction_loss: bool
    """Whether to use reconstruction loss for AE and VAE"""
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
    network_architecture: t.Literal["vanilla_cnn", "residual_network"]
    """Type of network architecture to use"""
    vanilla_cnn_config: t.Optional[VanillaCNNConfig]
    """
    Config for the vanilla cnn network architecture. Only used if 
    network_architecture is vanilla_cnn
    """

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

    #
    # PackNet
    #
    use_packnet: bool
    """Whether to use packnet"""
    prune_proportion: float
    """Proportion of the network to prune"""
    retrain_epochs: int
    """Number of epochs post-pruning to retrain the network"""
    task_inference_strategy: t.Literal["task_oracle", "task_reconstruction_loss"]
    """Type of task inference strategy to use"""

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

    def __init__(self) -> None:
        # Control variables held constant for all experiments
        self.batch_size = 64
        self.learning_rate = 0.0001
        self.device = "cuda"

    def configure_vanilla_cnn(self: 'ExperimentConfiguration') -> 'ExperimentConfiguration':
        """Configure the experiment to use a vanilla CNN"""
        self.latent_dims = 64
        self.network_architecture = "vanilla_cnn"
        self.vanilla_cnn_config = VanillaCNNConfig()
        self.vanilla_cnn_config.base_channels = 128
        return self

    def use_resnet(self: 'ExperimentConfiguration') -> 'ExperimentConfiguration':
        """Configure the experiment to use a ResNet CNN"""
        self.network_architecture = "residual_network"
        self.latent_dims = 64
        return self

    def fmnist(self: 'ExperimentConfiguration') -> 'ExperimentConfiguration':
        """Configure the experiment for the Fashion-MNIST dataset"""
        self.tensorboard_dir = "experiment_logs/fmnist"
        self.dataset_name = "FMNIST"
        self.dataset_root = "/Scratch/al183/datasets"
        self.input_channel_size = 1
        self.fixed_class_order = True
        self.is_image_data = True
        self.n_experiences = 5

        self.total_task_epochs = 20
        self.retrain_epochs = 5
        return self.configure_vanilla_cnn()

    def cifar10(self: 'ExperimentConfiguration') -> 'ExperimentConfiguration':
        """Configure the experiment for the CIFAR10 dataset"""
        self.tensorboard_dir = "experiment_logs/cifar10"
        self.dataset_name = "CIFAR10"
        self.dataset_root = "/Scratch/al183/datasets"
        self.input_channel_size = 3
        self.fixed_class_order = True
        self.is_image_data = True
        self.n_experiences = 5

        self.total_task_epochs = 50
        self.retrain_epochs = 10
        return self.use_resnet()

    def configure_ae(self: 'ExperimentConfiguration') -> 'ExperimentConfiguration':
        """Configure the experiment to use an AutoEncoder"""
        self.deep_generative_type = "AE"
        self.use_classifier_loss = True
        self.use_reconstruction_loss = True
        self.use_vae_loss = False
        self.classifier_loss_weight = 1.0
        self.reconstruction_loss_weight = 1.0
        return self

    def configure_vae(self: 'ExperimentConfiguration') -> 'ExperimentConfiguration':
        """Configure the experiment to use a variational AutoEncoder"""
        self.deep_generative_type = "VAE"
        self.use_classifier_loss = True
        self.use_reconstruction_loss = True
        self.use_vae_loss = True
        self.classifier_loss_weight = 1.0
        self.reconstruction_loss_weight = 1.0
        self.vae_loss_weight = 0.001
        return self

    def configure_packnet(self: 'ExperimentConfiguration') -> 'ExperimentConfiguration':
        """Configure the experiment to use PackNet"""
        self.use_packnet = True
        self.prune_proportion = 0.5
        self.task_inference_strategy = "task_oracle"
        return self

    def configure_transient(self: 'ExperimentConfiguration') -> 'ExperimentConfiguration':
        """Configure the experiment to not do any continual learning"""
        self.use_packnet = False
        self.n_experiences = 1
        return self
