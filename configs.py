from dataclasses import _MISSING_TYPE, dataclass, field

@dataclass
class DefaultConfig:
    save_path:str = 'results'
    pretrained_model: str = ''
    experiments_path:str = 'experiments/experiment.csv'
    seed:int = 1
    model_name: str = 'korean_tacotron2'
    process_name: str = 'tgt'
    

@dataclass
class SplitDataConfig(DefaultConfig):
    model_name: str = 'split_data'
    save_script_path:str = 'data'
    audio_path: str = ''
    script_path: str = ''
    test_size:int = 10


@dataclass
class Tacotron2Config(DefaultConfig):
    pretrained_model: str = ''

    #################### basic training params ###################################
    no_cuda: bool = False
    device: str = 'cuda'

    train_batch_size: int = 16
    eval_batch_size: int = 1
    gradient_accumulation_steps:int = 4
    steps_per_checkpoint:int = 5000
    steps_per_evaluate: int = 1000
    grad_clip_thresh:float = 1.0
    weight_decay:float = 1e-6
    learning_rate:float = 1e-3
    num_train_epochs:int = 500
    max_steps:int  = -1
    warmup_steps: int = -1
    warmup_percent:float = 0.0
    logging_steps:int = 5
    fp16_run:bool = False
    fp16_opt_level:str="O1"
    n_gpu:int=1
    local_rank:int=-1

    ## generator
    generator_path:str = 'checkpoints_g/vocgan_kss_pretrained_model_epoch_4500.pt'

    ########################## dataset options ###################################
    process_name:str = 'tgt'
    sampling_rate: int = 22050
    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mel_channels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0
    train_script: str = 'data/KSS/train.txt'
    val_script: str = 'data/KSS/dev.txt'
    load_mel_from_disk: bool = False
    normalize_option:str = 'NFKD'
    g2p_lib:str = 'g2pk'

    ############################ model params ######################################
    model_name: str = 'tacotron2'

    ## tokenizer
    num_labels: int = 1
    
    ## encoder
    n_symbols:int = 100
    symbols_embedding_dim:int = 512
    encoder_kernel_size:int = 5
    encoder_n_convolutions:int = 3
    encoder_embedding_dim:int = 512

    ## decoder
    n_frames_per_step:int = 1
    decoder_rnn_dim:int = 1024
    prenet_dim:int = 256
    max_decoder_steps:int = 1000
    gate_threshold:float = 0.5
    p_attention_dropout:float = 0.1
    p_decoder_dropout:float = 0.1

    # Attention parameters
    attention_rnn_dim:int = 1024
    attention_dim:int = 128

    # Location Layer parameters
    attention_location_n_filters:int = 32
    attention_location_kernel_size:int = 31

    # Mel-post processing network parameters
    postnet_embedding_dim:int = 512
    postnet_kernel_size:int = 5
    postnet_n_convolutions:int = 5
    
    ## loss
    mask_padding: bool = True

    


