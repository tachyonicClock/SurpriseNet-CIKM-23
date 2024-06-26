import copy
import json
import os
import typing as t
from operator import getitem, setitem


class ExpConfig:
    def __init__(self) -> None:
        # EXPERIMENT METADATA
        self.label: str
        """Label of an experiment given by the user, probably shouldn't use '_'
        because they are used to split the label into the experiment name.
        """
        self.name: str
        """Long name of the experiment used to track the repository hash,
        label, scenario, strategy, and architecture.
        """
        self.tensorboard_dir: str = "experiment_logs"
        """Directory to store tensorboard logs. This is relative to the root of the
        project.
        """
        self.repo_hash: t.Optional[str] = None
        """Hash of the repository at the time of the experiment. This ensures
        that the experiment is reproducible. This will be marked as dirty
        if there are uncommitted changes.
        """

        # SCENARIO
        self.dataset_name: str
        """Short code to select the dataset e.g CIFAR10"""
        self.scenario_name: str
        """Short code to select the correct scenario e.g S-FMNIST"""
        self.dataset_root: t.Optional[str] = os.environ.get("DATASETS", None)
        """Where datasets should be accessed from or downloaded to"""
        self.fixed_class_order: t.Optional[t.List[int]] = None
        """A fixed class order to use for the scenario. Randomized if None"""
        self.n_experiences: int
        """Number of experiences/tasks in the scenario"""
        self.input_shape: t.Tuple[int, int, int]
        """Dimensions of inputs to the network"""
        self.is_image_data: bool
        """Does attempting to display the output of the network make sense?"""
        self.n_classes: int
        """Number of classes in the dataset"""
        self.normalize: bool = False
        """Should the data be normalized? We normalize only SE-CIFAR100 and
        SE-CORe50 because they use pre-trained frozen feature extractors, that
        expect normalized data. We don't normalize the other datasets because
        they are reconstructed using BCE loss, which expects data scaled to
        between 0 and 1.
        """

        # LOSS
        self.classifier_loss_type: t.Literal["CrossEntropy", "LogitNorm"] = (
            "CrossEntropy"
        )
        """Type of loss function to use"""
        self.classifier_loss_kwargs: t.Dict[str, t.Any] = {}
        """Kwargs to pass to the classifier loss"""
        self.classifier_loss_weight: t.Optional[float] = 1.0
        """Weight of the classifier loss. None if classifier loss is not used"""
        self.reconstruction_loss_type: t.Literal["mse", "bce"] = "bce"
        """Type of loss function to use"""
        self.reconstruction_loss_weight: t.Optional[float] = None
        """Weight of the reconstruction loss. None if reconstruction loss is
        not used"""
        self.vae_loss_weight: t.Optional[float] = None
        """Weight of the VAE loss or Kullback-Leibler divergence strength. 
        None if VAE loss is not used"""

        # ARCHITECTURE
        self.latent_dims: int
        """Number of latent dimensions of the VAE/AE"""
        self.architecture: t.Literal["AE", "VAE", "DeepVAE"]
        """Type of auto-encoder to use"""
        self.network_style: t.Literal[
            "vanilla_cnn", "residual", "mlp", "DeepVAE_FMNIST", "DeepVAE_CIFAR"
        ]
        """Type of network to be used"""
        self.embedding_module: t.Literal[
            "None", "ResNet50", "SmallResNet18", "ResNet18"
        ] = "None"
        """Optionally configure the experiment to embed the dataset"""
        self.network_cfg: t.Dict[str, t.Any] = {}
        """Other network configuration options"""
        self.pretrained_root: t.Optional[str] = "pretrained"
        """Where to store and retrieve pretrained models from"""

        # TRAINING
        self.total_task_epochs: int
        """Total number of training epochs in a task"""
        self.batch_size: int = 64
        """Batch size"""
        self.learning_rate: float = 0.0001
        """Learning rate"""
        self.device: str = "cuda"
        """Device to use for training"""

        # PackNet and SurpriseNet specific
        self.use_packnet: bool = False
        """Whether to use the packnet training procedure"""
        self.prune_proportion: t.Union[float, t.List[float]] = 0.5
        """Proportion of the network to prune"""
        self.retrain_epochs: int
        """Number of epochs post-pruning to retrain the network"""
        self.task_inference_strategy: t.Literal[
            "task_oracle", "task_reconstruction_loss"
        ]
        """Type of task inference strategy to use"""
        self.task_inference_strategy_kwargs: t.Optional[t.Dict[str, t.Any]] = None
        self.optimizer: t.Literal["Adam", "SGD"] = "Adam"

    def toJSON(self):
        return json.dumps(self.__dict__, indent=4)

    #
    # Networks
    #

    def _network_cnn(self: "ExpConfig") -> "ExpConfig":
        """Configure the experiment to use a vanilla CNN"""
        self.network_style = "vanilla_cnn"
        self.network_cfg["base_channels"] = 128
        return self

    def _network_mlp(self) -> "ExpConfig":
        """Configure the experiment to use a rectangular network"""
        self.network_style = "mlp"
        self.network_cfg["width"] = 512
        return self

    def _network_resnet(self: "ExpConfig") -> "ExpConfig":
        """Configure the experiment to use a ResNet CNN"""
        self.network_style = "residual"
        return self

    #
    # Scenarios
    #

    def scenario_fmnist(self: "ExpConfig") -> "ExpConfig":
        """Configure the experiment for the Fashion-MNIST dataset"""
        self._network_cnn()
        self.dataset_name = "FMNIST"
        self.input_shape = (1, 32, 32)
        self.is_image_data = True
        self.latent_dims = 64
        self.n_classes = 10
        self.n_experiences = 5
        self.retrain_epochs = 5
        self.total_task_epochs = 20
        # self.mask_classifier_loss = True
        return self

    def scenario_cifar10(self: "ExpConfig") -> "ExpConfig":
        """Configure the experiment for the CIFAR10 dataset"""
        self._network_resnet()
        self.dataset_name = "CIFAR10"
        self.input_shape = (3, 32, 32)
        self.is_image_data = True
        self.latent_dims = 128
        self.n_classes = 10
        self.n_experiences = 5
        self.retrain_epochs = 10
        self.total_task_epochs = 50
        return self

    def scenario_cifar100(self) -> "ExpConfig":
        """Configure the experiment for the CIFAR100 dataset"""
        self._network_resnet()
        self.dataset_name = "CIFAR100"
        self.input_shape = (3, 32, 32)
        self.is_image_data = True
        self.latent_dims = 256
        self.n_classes = 100
        self.n_experiences = 10
        self.retrain_epochs = 30
        self.total_task_epochs = 100
        return self

    def scenario_core50(self: "ExpConfig") -> "ExpConfig":
        """Configure the experiment for the Core50 dataset"""
        self._network_resnet()
        self.dataset_name = "M_CORe50_NC"
        self.input_shape = (3, 32, 32)
        self.is_image_data = True
        self.latent_dims = 512
        self.n_classes = 50
        self.n_experiences = 10
        self.retrain_epochs = 25
        self.total_task_epochs = 50
        return self

    def scenario_embedded_fmnist(self: "ExpConfig") -> "ExpConfig":
        # Dataset configuration
        self.dataset_name = "FMNIST"
        self.n_classes = 10
        self.n_experiences = 5
        self.embedding_module = "ResNet18"
        self.normalize = True
        self.is_image_data = False

        # Network configuration
        self.network_style = "mlp"
        self.latent_dims = 128
        self.input_shape = (512,)
        self.network_cfg["width"] = 512
        self.network_cfg["layer_count"] = 5
        self.network_cfg["layer_growth"] = 1.0
        self.network_cfg["dropout"] = 0.1

        # Training configuration
        self.batch_size = 128
        self.reconstruction_loss_type = "mse"
        self.retrain_epochs = 5
        self.total_task_epochs = 20
        self.learning_rate = 0.0008
        return self

    def scenario_embedded_cifar10(self: "ExpConfig") -> "ExpConfig":
        # Dataset configuration
        self.dataset_name = "CIFAR10"
        self.n_classes = 10
        self.n_experiences = 5
        self.embedding_module = "ResNet18"
        self.normalize = True
        self.is_image_data = False

        # Network configuration
        self.network_style = "mlp"
        self.latent_dims = 128
        self.input_shape = (512,)
        self.network_cfg["width"] = 512
        self.network_cfg["layer_count"] = 5
        self.network_cfg["layer_growth"] = 1.0
        self.network_cfg["dropout"] = 0.1

        # Training configuration
        self.batch_size = 128
        self.reconstruction_loss_type = "mse"
        self.retrain_epochs = 25
        self.total_task_epochs = 50
        self.learning_rate = 0.0008
        return self

    def scenario_embedded_cifar100(self: "ExpConfig") -> "ExpConfig":
        """Configure the experiment to use the embedded CIFAR100 dataset"""
        self._network_mlp()
        self.dataset_name = "CIFAR100"
        self.embedding_module = "ResNet18"
        self.is_image_data = False
        self.n_classes = 100
        self.n_experiences = 10
        self.prune_proportion = 0.5
        self.reconstruction_loss_type = "mse"
        self.normalize = True
        self.retrain_epochs = 30
        self.total_task_epochs = 100

        # Network configuration
        self.network_style = "mlp"
        self.latent_dims = 128
        self.input_shape = (512,)
        self.network_cfg["width"] = 512
        self.network_cfg["layer_count"] = 5
        self.network_cfg["layer_growth"] = 1.0
        self.network_cfg["dropout"] = 0.1
        self.learning_rate = 0.0008
        return self

    def scenario_embedded_core50(self: "ExpConfig") -> "ExpConfig":
        """Configure the experiment to use the embedded CIFAR100 dataset"""
        self._network_mlp()
        self.dataset_name = "CORe50_NC"
        self.embedding_module = "ResNet18"
        self.is_image_data = False
        self.n_classes = 50
        self.n_experiences = 10
        self.prune_proportion = 0.5
        self.reconstruction_loss_type = "mse"
        self.normalize = True
        self.retrain_epochs = 25
        self.total_task_epochs = 50

        # Network configuration
        self.network_style = "mlp"
        self.latent_dims = 128
        self.input_shape = (512,)
        self.network_cfg["width"] = 512
        self.network_cfg["layer_count"] = 5
        self.network_cfg["layer_growth"] = 1.0
        self.network_cfg["dropout"] = 0.1
        self.learning_rate = 0.0008
        return self

    def scenario_dsads(self) -> "ExpConfig":
        self._network_mlp()
        self.batch_size = 500
        self.latent_dims = 128
        self.network_cfg["width"] = 512
        self.network_cfg["layer_count"] = 5
        self.network_cfg["layer_growth"] = 1.0
        self.network_cfg["dropout"] = 0.1
        self.learning_rate = 0.0008
        self.dataset_name = "DSADS"
        self.input_shape = (405,)
        self.is_image_data = False
        self.n_classes = 19
        self.n_experiences = 9
        self.total_task_epochs = 200
        self.retrain_epochs = 100
        self.reconstruction_loss_type = "mse"
        return self

    def scenario_pamap2(self) -> "ExpConfig":
        self._network_mlp()
        self.batch_size = 500
        self.network_cfg["width"] = 256
        self.latent_dims = 64
        self.network_cfg["layer_count"] = 5
        self.network_cfg["layer_growth"] = 1.0
        self.network_cfg["dropout"] = 0.1
        self.learning_rate = 0.0008
        self.dataset_name = "PAMAP2"
        self.input_shape = (243,)
        self.is_image_data = False
        self.n_classes = 12
        self.n_experiences = 6
        self.total_task_epochs = 200
        self.retrain_epochs = 100
        self.reconstruction_loss_type = "mse"
        return self

    def arch_autoencoder(self: "ExpConfig") -> "ExpConfig":
        """Configure the experiment to use an AutoEncoder"""
        self.architecture = "AE"
        self.reconstruction_loss_weight = 1.0
        self.use_vae_loss = False
        return self

    def arch_variational_auto_encoder(self: "ExpConfig") -> "ExpConfig":
        """Configure the experiment to use a variational AutoEncoder"""
        self.architecture = "VAE"
        self.reconstruction_loss_weight = 1.0
        self.vae_loss_weight = 0.001
        return self

    def strategy_packnet(self: "ExpConfig") -> "ExpConfig":
        """Configure the experiment to use PackNet"""
        self.use_packnet = True
        self.task_inference_strategy = "task_oracle"
        return self

    def strategy_surprisenet(self: "ExpConfig") -> "ExpConfig":
        """Configure the experiment to use CI-PackNet"""
        self.use_packnet = True

        if self.architecture == "DeepVAE":
            self.task_inference_strategy = "log_likelihood_ratio"
            self.task_inference_strategy_kwargs = dict(k=1)
        else:
            self.task_inference_strategy = "task_reconstruction_loss"
        return self

    def strategy_not_cl(self: "ExpConfig") -> "ExpConfig":
        """Configure the experiment to not do any continual learning"""
        self.n_experiences = 1
        return self

    def strategy_replay(self: "ExpConfig") -> "ExpConfig":
        """Configure the experiment to use replay"""
        self.use_experience_replay = True
        return self

    def copy(self: "ExpConfig") -> "ExpConfig":
        """Create a copy of the experiment configuration"""
        return copy.deepcopy(self)

    @staticmethod
    def from_json(json_file: str) -> "ExpConfig":
        """Create an experiment configuration from a JSON file"""
        with open(json_file, "r") as f:
            json_cfg = json.load(f)
        cfg = ExpConfig()
        # Move the keys from the dictionary to the config
        for key, value in json_cfg.items():
            setattr(cfg, key, value)

        return cfg

    def set_dotpath(self, dotpath: str, value: t.Any) -> "ExpConfig":
        """Set a value using a dotpath"""
        dotpath = dotpath.split(".")
        obj = self
        for key in dotpath[:-1]:
            if hasattr(obj, "__getitem__"):
                obj = getitem(obj, key)
            else:
                obj = getattr(obj, key)

        if hasattr(obj, "__setitem__"):
            setitem(obj, dotpath[-1], value)
        else:
            setattr(obj, dotpath[-1], value)
        return self

    def get_dotpath(self, dotpath: str) -> t.Any:
        """Get a value using a dotpath"""
        dotpath = dotpath.split(".")
        obj = self
        for key in dotpath:
            if hasattr(obj, "__getitem__"):
                obj = getitem(obj, key)
            else:
                obj = getattr(obj, key)
        return obj

    def set_task_order_from_txt(self, class_order_filename: str) -> "ExpConfig":
        task_comp = []
        with open(class_order_filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                task_comp.extend(line.split(","))
        self.fixed_class_order = list(map(int, task_comp))
        return self
