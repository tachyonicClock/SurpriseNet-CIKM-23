import copy
import json
import os
import typing as t


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

        # GAUSSIAN SCHEDULE
        self.task_free = False
        """Should a task-free scenario be used?"""
        self.task_free_instances_in_task: t.Optional[int] = None
        """In a task-free scenario, how many instances should be used per task"""
        self.task_free_width: float = 1 / 20
        """In a task-free scenario, how wide should the gaussian be"""
        self.test_every: int = 1
        """Limit how often the test set is evaluated. This is useful for 
        microtasks where there are lots of tasks and the test set is relatively
        large.
        """
        # DRIFT DETECTION
        self.task_free_drift_detector: t.Optional[t.Literal["clock_oracle"]] = None
        """If the scenario is task-free, what drift detector should be used?"""
        self.task_free_drift_detector_kwargs: t.Optional[t.Dict[str, t.Any]] = None
        """If the scenario is task-free, what kwargs should be passed to the
        drift detector?"""

        # LOSS
        self.classifier_loss_weight: t.Optional[float] = 1.0
        """Weight of the classifier loss. None if classifier loss is not used"""
        self.reconstruction_loss_type: t.Literal["mse", "bce", "DeepVAE_ELBO"] = "bce"
        """Type of loss function to use"""
        self.reconstruction_loss_weight: t.Optional[float] = None
        """Weight of the reconstruction loss. None if reconstruction loss is
        not used"""
        self.vae_loss_weight: t.Optional[float] = None
        """Weight of the VAE loss or Kullback-Leibler divergence strength. 
        None if VAE loss is not used"""
        self.mask_classifier_loss = False
        self.HVAE_schedule: t.Optional[dict] = None

        # ARCHITECTURE
        self.latent_dims: int
        """Number of latent dimensions of the VAE/AE"""
        self.architecture: t.Literal["AE", "VAE", "DeepVAE"]
        """Type of auto-encoder to use"""
        self.network_style: t.Literal[
            "vanilla_cnn", "residual", "mlp", "DeepVAE_FMNIST"
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
            "task_oracle", "task_reconstruction_loss", "log_likelihood_ratio"
        ]
        """Type of task inference strategy to use"""
        self.task_inference_strategy_kwargs: t.Optional[t.Dict[str, t.Any]] = None

        # OTHER STRATEGIES
        # Experience replay
        self.replay_buffer: t.Optional[int] = None
        """Number of instances to store in the replay buffer. If 0 or None, no
        replay buffer is used.
        """
        # Synaptic intelligence
        self.si_lambda: t.Optional[float] = None
        """Synaptic Intelligence Lambda. If 0 or None, no
        synaptic intelligence is used."""

        self.lwf_alpha: t.Optional[float] = None
        """Learning without Forgetting alpha. If None LWF is not used"""

        self.log_mini_batch: bool = False
        """Log the loss and accuracy of each minibatch"""

        self.cumulative: bool = False

        # CONTINUAL HYPERPARAMETER FRAMEWORK
        self.continual_hyperparameter_framework: bool = False
        """Whether to use the continual hyperparameter framework"""
        self.chf_validation_split_proportion: float = 0.2
        """What fraction of the experience should be used for validation"""
        self.chf_lr_grid = [0.00005, 0.0001, 0.0005, 0.001]
        """Learning rate grid to search over during maximal plasticity search"""
        self.chf_accuracy_drop_threshold = 0.1
        """Threshold for acceptable accuracy drop"""
        self.chf_stability_decay = 0.9
        """How quickly the stability decays during stability decay search"""

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
        self.batch_size = 128
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
        self.retrain_epochs = 2
        self.total_task_epochs = 10
        return self

    def scenario_embedded_cifar100(self: "ExpConfig") -> "ExpConfig":
        """Configure the experiment to use the embedded CIFAR100 dataset"""
        self._network_mlp()
        self.dataset_name = "CIFAR100"
        self.embedding_module = "ResNet18"
        self.input_shape = (512,)
        self.is_image_data = False
        self.latent_dims = 256
        self.n_classes = 100
        self.n_experiences = 10
        self.prune_proportion = 0.5
        self.reconstruction_loss_type = "mse"
        self.normalize = True
        self.retrain_epochs = 30
        self.total_task_epochs = 100
        return self

    def scenario_embedded_core50(self: "ExpConfig") -> "ExpConfig":
        """Configure the experiment to use the embedded CIFAR100 dataset"""
        self._network_mlp()
        self.dataset_name = "CORe50_NC"
        self.embedding_module = "ResNet18"
        self.input_shape = (512,)
        self.is_image_data = False
        self.latent_dims = 512
        self.n_classes = 50
        self.n_experiences = 10
        self.prune_proportion = 0.5
        self.reconstruction_loss_type = "mse"
        self.normalize = True
        self.retrain_epochs = 2
        self.total_task_epochs = 10
        return self

    def scenario_gaussian_schedule_mnist(self: "ExpConfig") -> "ExpConfig":
        """Configure the experiment for the Fashion-MNIST dataset with a
        Gaussian schedule"""
        self._network_cnn()
        # self.learning_rate = 0.01
        self.dataset_name = "MNIST"
        self.input_shape = (1, 32, 32)
        self.is_image_data = True
        self.latent_dims = 4
        self.n_classes = 10

        # self.mask_classifier_loss = True
        self.classifier_loss_weight = None

        # Setup the schedule
        self.n_experiences = 200
        self.task_free = True
        self.task_free_instances_in_task = self.batch_size * 10
        self.task_free_width = 1 / 20

        self.test_every = self.n_experiences / self.n_classes
        self.retrain_epochs = 0
        self.total_task_epochs = 1

        self.task_free_drift_detector = "clock_oracle"
        self.task_free_drift_detector_kwargs = dict(
            drift_period=self.n_experiences / self.n_classes,
            warn_in_advance=5,
            drift_in_advance=1,
        )

        self.optimizer = "Adam"
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

    def arch_deep_vae(self: "ExpConfig") -> "ExpConfig":
        """Configure the experiment to use a deep VAE"""
        self.architecture = "DeepVAE"
        self.reconstruction_loss_type = "DeepVAE_ELBO"
        self.reconstruction_loss_weight = 1.0
        self.classifier_loss_weight = 1.0
        self.HVAE_schedule = {"warmup_epochs": 100, "enable_free_nats": False}
        self.total_task_epochs = 200
        self.retrain_epochs = 50
        self.latent_dims = 8

        if self.dataset_name == "FMNIST":
            self.batch_size = 256
            self.total_task_epochs = 180
            self.retrain_epochs = 45

            self.network_style = "DeepVAE_FMNIST"
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
