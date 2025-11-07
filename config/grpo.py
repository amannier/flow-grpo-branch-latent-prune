import ml_collections
import imp
import os
from datetime import datetime  # 新增导入

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config

def general_ocr_wan2_1():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # config.pretrained.model = "hf_cache/Wan2.1-T2V-14B-Diffusers"
    config.pretrained.model = "hf_cache/Wan2.1-T2V-1.3B-Diffusers"
    config.sample.num_steps = 20
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale=4.5
    config.run_name = "wan_flow_grpo"
    
    config.height = 240
    config.width = 416
    config.frames = 33
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 4 # 12
    config.sample.num_batches_per_epoch = 2
    config.sample.sample_time_per_prompt = 1
    config.sample.test_batch_size = 2

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch * config.sample.sample_time_per_prompt // 2 if (config.sample.num_batches_per_epoch * config.sample.sample_time_per_prompt) > 1 else 1
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.004
    config.train.learning_rate = 1e-4
    config.train.clip_range=1e-3
    # kl reward
    # KL reward and KL loss are two ways to incorporate KL divergence. KL reward adds KL to the reward, while KL loss, introduced by GRPO, directly adds KL loss to the policy loss. We support both methods, but KL loss is recommended as the preferred option.
    config.sample.kl_reward = 0
    # We also support using SFT data in RL training for supervised learning to prevent quality drop, but this option was unused
    config.train.sft=0.0
    config.train.sft_batch_size=3
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std=False
    config.train.ema=True
    config.mixed_precision = "bf16"
    config.diffusion_loss = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.num_epochs = 100000
    config.save_freq = 60 # epoch
    config.eval_freq = 30
    config.save_dir = f'logs/video_ocr/{config.run_name}'
    config.resume_from = None
    config.reward_fn = {
        "video_ocr": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def general_ocr_sd3():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    # Whether to use the same noise for the same prompt
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/sd3.5-M'
    config.reward_fn = {
        "ocr": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def geneval_sd3():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.04
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = f'logs/geneval/sd3.5-M'
    config.reward_fn = {
        "geneval": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config



def geneval_sd3_new_ori():
    gpu_number = 24
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.04
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.run_name = f'geneval_ori-24_bs_8'
    config.save_dir = f'logs/geneval/{config.run_name}_{timestamp}'
    config.reward_fn = {
        "geneval": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config


def geneval_sd3_new_ori_bs_4():
    gpu_number = 24
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.04
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.run_name = f'geneval_ori-24_bs_4'
    config.save_dir = f'logs/geneval/{config.run_name}_{timestamp}'
    config.reward_fn = {
        "geneval": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config



def geneval_sd3_new_48_to_12():
    gpu_number = 24
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt_before_pruning = 48
    config.sample.skip_timesteps = [5,7]
    config.sample.lefts = [24,12] 
    config.sample.num_image_per_prompt = config.sample.lefts[-1]  
    # 所有进程一次batch总共算了多少不同的prompt
    config.sample.unique_prompts = int(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning)
    # 加一个sample_batches_per_epoch，用于记录每次采样多少个batch，因为每次采样多少个batch是根据num_image_per_prompt_before_pruning计算的，所以需要一个变量来记录
    config.sample.sample_batches_per_epoch = int(96/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning))
    config.sample.num_batches_per_epoch = int(96/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.04
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.run_name = f'geneval_48_to_12_bs_8'
    config.save_dir = f'logs/geneval/{config.run_name}_{timestamp}'
    config.reward_fn = {
        "geneval": 1.0,
    }
    config.prompt_fn = "geneval"
    config.per_prompt_stat_tracking = True
    return config


def geneval_sd3_new_24_to_12():
    gpu_number = 24
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt_before_pruning = 24
    config.sample.skip_timesteps = [5,7]
    config.sample.lefts = [16,12] 
    config.sample.num_image_per_prompt = config.sample.lefts[-1]  
    # 所有进程一次batch总共算了多少不同的prompt
    config.sample.unique_prompts = int(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning)
    # 加一个sample_batches_per_epoch，用于记录每次采样多少个batch，因为每次采样多少个batch是根据num_image_per_prompt_before_pruning计算的，所以需要一个变量来记录
    config.sample.sample_batches_per_epoch = int(96/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning))
    config.sample.num_batches_per_epoch = int(96/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.04
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.run_name = f'geneval_24_to_12_bs_8'
    config.save_dir = f'logs/geneval/{config.run_name}_{timestamp}'
    config.reward_fn = {
        "geneval": 1.0,
    }
    config.prompt_fn = "geneval"
    config.per_prompt_stat_tracking = True
    return config


def geneval_sd3_new_48_to_12_bs_4():
    gpu_number = 24
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt_before_pruning = 48
    config.sample.skip_timesteps = [5,7]
    config.sample.lefts = [24,12] 
    config.sample.num_image_per_prompt = config.sample.lefts[-1]  
    # 所有进程一次batch总共算了多少不同的prompt
    config.sample.unique_prompts = int(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning)
    # 加一个sample_batches_per_epoch，用于记录每次采样多少个batch，因为每次采样多少个batch是根据num_image_per_prompt_before_pruning计算的，所以需要一个变量来记录
    config.sample.sample_batches_per_epoch = int(96/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning))
    config.sample.num_batches_per_epoch = int(96/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.04
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.run_name = f'geneval_48_to_12_bs_4'
    config.save_dir = f'logs/geneval/{config.run_name}_{timestamp}'
    config.reward_fn = {
        "geneval": 1.0,
    }
    config.prompt_fn = "geneval"
    config.per_prompt_stat_tracking = True
    return config


def geneval_sd3_new_48_to_12_bs_4_48_gpu():
    gpu_number = 48
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt_before_pruning = 48
    config.sample.skip_timesteps = [5,7]
    config.sample.lefts = [24,12] 
    config.sample.num_image_per_prompt = config.sample.lefts[-1]  
    # 所有进程一次batch总共算了多少不同的prompt
    config.sample.unique_prompts = int(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning)
    # 加一个sample_batches_per_epoch，用于记录每次采样多少个batch，因为每次采样多少个batch是根据num_image_per_prompt_before_pruning计算的，所以需要一个变量来记录
    config.sample.sample_batches_per_epoch = int(96/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning))
    config.sample.num_batches_per_epoch = int(96/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.04
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.run_name = f'geneval_48_to_12_bs_4_48_gpu'
    config.save_dir = f'logs/geneval/{config.run_name}_{timestamp}'
    config.reward_fn = {
        "geneval": 1.0,
    }
    config.prompt_fn = "geneval"
    config.per_prompt_stat_tracking = True
    return config


def geneval_sd3_new_48_to_16_bs_3_48_gpu():
    gpu_number = 48
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt_before_pruning = 48
    config.sample.skip_timesteps = [5,7]
    config.sample.lefts = [24,16] 
    config.sample.num_image_per_prompt = config.sample.lefts[-1]  
    # 所有进程一次batch总共算了多少不同的prompt
    config.sample.unique_prompts = int(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning)
    # 加一个sample_batches_per_epoch，用于记录每次采样多少个batch，因为每次采样多少个batch是根据num_image_per_prompt_before_pruning计算的，所以需要一个变量来记录
    config.sample.sample_batches_per_epoch = int(72/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning))
    config.sample.num_batches_per_epoch = int(72/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.04
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.run_name = f'geneval_48_to_16_bs_3_48_gpu'
    config.save_dir = f'logs/geneval/{config.run_name}_{timestamp}'
    config.reward_fn = {
        "geneval": 1.0,
    }
    config.prompt_fn = "geneval"
    config.per_prompt_stat_tracking = True
    return config


def geneval_sd3_new_24_to_12_bs_4():
    gpu_number = 24
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt_before_pruning = 24
    config.sample.skip_timesteps = [5,7]
    config.sample.lefts = [16,12] 
    config.sample.num_image_per_prompt = config.sample.lefts[-1]  
    # 所有进程一次batch总共算了多少不同的prompt
    config.sample.unique_prompts = int(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning)
    # 加一个sample_batches_per_epoch，用于记录每次采样多少个batch，因为每次采样多少个batch是根据num_image_per_prompt_before_pruning计算的，所以需要一个变量来记录
    config.sample.sample_batches_per_epoch = int(96/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning))
    config.sample.num_batches_per_epoch = int(96/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.04
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.run_name = f'geneval_24_to_12_bs_4'
    config.save_dir = f'logs/geneval/{config.run_name}_{timestamp}'
    config.reward_fn = {
        "geneval": 1.0,
    }
    config.prompt_fn = "geneval"
    config.per_prompt_stat_tracking = True
    return config


def geneval_sd3_ori_2gpu():
    gpu_number = 2
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "/data11/xinyue.liu/sjy/TempFlow-GRPO/base_models/models--stabilityai--stable-diffusion-3.5-medium/b940f670f0eda2d07fbb75229e779da1ad11eb80"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(8/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.04
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    # 新增：生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 更新：将时间戳附加到 save_dir
    config.run_name = f'geneval_ori-16'
    config.save_dir = f'logs/geneval/{config.run_name}_{timestamp}'
    # config.save_dir = f'logs/geneval/sd3.5-M'
    config.reward_fn = {
        "geneval": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config

def pickscore_sd3():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    # 新增：生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 更新：将时间戳附加到 save_dir
    config.save_dir = f'logs/pickscore/sd3.5-M-origin-24_{timestamp}'
    # config.save_dir = 'logs/pickscore/sd3.5-M'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def pickscore_sd3_pruning():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt_before_pruning = 32
    config.sample.num_image_per_prompt = 16
    config.sample.unique_prompts = int(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning) # 6
    # 加一个sample_batches_per_epoch，用于记录每次采样多少个batch，因为每次采样多少个batch是根据num_image_per_prompt_before_pruning计算的，所以需要一个变量来记录
    config.sample.sample_batches_per_epoch = int(64/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning))
    config.sample.num_batches_per_epoch = int(64/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    # 新增：生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 更新：将时间戳附加到 save_dir
    config.save_dir = f'logs/pickscore/sd3.5-M-latent-pruning-48-to-24_{timestamp}'
    # config.save_dir = 'logs/pickscore/sd3.5-M'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def pickscore_sd3_pruning_new_48_to_24():
    gpu_number = 24
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.pretrained.reward_model = 'pickscore'
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt_before_pruning = 48
    config.sample.skip_timesteps = [5,7]
    config.sample.lefts = [32,24]
    config.sample.num_image_per_prompt = config.sample.lefts[-1]
    # 所有进程一次batch总共算了多少不同的prompt
    config.sample.unique_prompts = int(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning)
    # 加一个sample_batches_per_epoch，用于记录每次采样多少个batch，因为每次采样多少个batch是根据num_image_per_prompt_before_pruning计算的，所以需要一个变量来记录
    config.sample.sample_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning))
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    # 新增：生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 更新：将时间戳附加到 save_dir
    config.run_name = f'pruning-48-to-24_{timestamp}'
    config.save_dir = f'logs/pickscore/{config.run_name}'
    # config.save_dir = 'logs/pickscore/sd3.5-M'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def pickscore_sd3_pruning_new_48_to_12():
    gpu_number = 24
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.pretrained.reward_model = 'pickscore'
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt_before_pruning = 48
    config.sample.skip_timesteps = [5,7]
    config.sample.lefts = [24,12]
    config.sample.num_image_per_prompt = config.sample.lefts[-1]
    # 所有进程一次batch总共算了多少不同的prompt
    config.sample.unique_prompts = int(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning)
    # 加一个sample_batches_per_epoch，用于记录每次采样多少个batch，因为每次采样多少个batch是根据num_image_per_prompt_before_pruning计算的，所以需要一个变量来记录
    config.sample.sample_batches_per_epoch = int(96/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning))
    config.sample.num_batches_per_epoch = int(96/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    # 新增：生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.run_name = f'pruning-48-to-12-bs-8'
    # 更新：将时间戳附加到 save_dir
    config.save_dir = f'logs/pickscore/{config.run_name}_{timestamp}'
    # config.save_dir = 'logs/pickscore/sd3.5-M'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def pickscore_sd3_pruning_new_24_to_12():
    gpu_number = 24
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.pretrained.reward_model = 'pickscore'
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt_before_pruning = 24
    config.sample.skip_timesteps = [5,7]
    config.sample.lefts = [16,12]
    config.sample.num_image_per_prompt = config.sample.lefts[-1]
    # 所有进程一次batch总共算了多少不同的prompt
    config.sample.unique_prompts = int(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning)
    # 加一个sample_batches_per_epoch，用于记录每次采样多少个batch，因为每次采样多少个batch是根据num_image_per_prompt_before_pruning计算的，所以需要一个变量来记录
    config.sample.sample_batches_per_epoch = int(96/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning))
    config.sample.num_batches_per_epoch = int(96/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    # 新增：生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.run_name = f'pruning-24-to-12-bs-8'
    # 更新：将时间戳附加到 save_dir
    config.save_dir = f'logs/pickscore/sd3.5-M-pruning-24-to-12-bs-8_{timestamp}'
    # config.save_dir = 'logs/pickscore/sd3.5-M'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def pickscore_sd3_new_origin_24():
    gpu_number=24
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    # 新增：生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 更新：将时间戳附加到 save_dir
    config.save_dir = f'logs/pickscore/sd3.5-M-origin-24_{timestamp}'
    # config.save_dir = 'logs/pickscore/sd3.5-M'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config



def pickscore_sd3_pruning_48_to_24_bs_4():
    gpu_number = 24
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.pretrained.reward_model = 'pickscore'
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt_before_pruning = 48
    config.sample.skip_timesteps = [5,7]
    config.sample.lefts = [32,24]
    config.sample.num_image_per_prompt = config.sample.lefts[-1]
    # 所有进程一次batch总共算了多少不同的prompt
    config.sample.unique_prompts = int(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning)
    # 加一个sample_batches_per_epoch，用于记录每次采样多少个batch，因为每次采样多少个batch是根据num_image_per_prompt_before_pruning计算的，所以需要一个变量来记录
    config.sample.sample_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning))
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    # 新增：生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.run_name = f'pruning-48-to-24-bs-4'
    # 更新：将时间戳附加到 save_dir
    config.save_dir = f'logs/pickscore/sd3.5-M-pruning-48-to-24-bs-4_{timestamp}'
    # config.save_dir = 'logs/pickscore/sd3.5-M'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config




def pickscore_sd3_pruning_48_to_12_bs_4():
    gpu_number = 24
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.pretrained.reward_model = 'pickscore'
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt_before_pruning = 48
    config.sample.skip_timesteps = [5,7]
    config.sample.lefts = [24,12]
    config.sample.num_image_per_prompt = config.sample.lefts[-1]
    # 所有进程一次batch总共算了多少不同的prompt
    config.sample.unique_prompts = int(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning)
    # 加一个sample_batches_per_epoch，用于记录每次采样多少个batch，因为每次采样多少个batch是根据num_image_per_prompt_before_pruning计算的，所以需要一个变量来记录
    config.sample.sample_batches_per_epoch = int(96/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning))
    config.sample.num_batches_per_epoch = int(96/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    # 新增：生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.run_name = f'pruning-48-to-12-bs-4'
    # 更新：将时间戳附加到 save_dir
    config.save_dir = f'logs/pickscore/sd3.5-M-pruning-48-to-12-bs-4_{timestamp}'
    # config.save_dir = 'logs/pickscore/sd3.5-M'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def pickscore_sd3_pruning_24_to_12_bs_4():
    gpu_number = 24
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.pretrained.reward_model = 'pickscore'
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt_before_pruning = 24
    config.sample.skip_timesteps = [5,7]
    config.sample.lefts = [16,12]
    config.sample.num_image_per_prompt = config.sample.lefts[-1]
    # 所有进程一次batch总共算了多少不同的prompt
    config.sample.unique_prompts = int(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning)
    # 加一个sample_batches_per_epoch，用于记录每次采样多少个batch，因为每次采样多少个batch是根据num_image_per_prompt_before_pruning计算的，所以需要一个变量来记录
    config.sample.sample_batches_per_epoch = int(96/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning))
    config.sample.num_batches_per_epoch = int(96/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    # 新增：生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.run_name = f'pruning-24-to-12-bs-4'
    # 更新：将时间戳附加到 save_dir
    config.save_dir = f'logs/pickscore/sd3.5-M-pruning-24-to-12-bs-4_{timestamp}'
    # config.save_dir = 'logs/pickscore/sd3.5-M'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config



def pickscore_sd3_new_origin_24_new_align_bs_4():
    gpu_number=24
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    # 新增：生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 更新：将时间戳附加到 save_dir
    config.run_name = f'origin-24-new-align-bs-4'
    config.save_dir = f'logs/pickscore/sd3.5-M-origin-24_new_align_bs_4_{timestamp}'
    # config.save_dir = 'logs/pickscore/sd3.5-M'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def pickscore_sd3_origin_12_bad_bs_8():
    gpu_number=24
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 12
    config.sample.num_batches_per_epoch = int(96/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    # 新增：生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 更新：将时间戳附加到 save_dir
    config.run_name = f'origin-12-bad-bs-8'
    config.save_dir = f'logs/pickscore/sd3.5-M-origin-12-bs_8_{timestamp}'
    # config.save_dir = 'logs/pickscore/sd3.5-M'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config



def pickscore_sd3_pruning_2gpu():
    gpu_number = 2
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "/data11/xinyue.liu/sjy/TempFlow-GRPO/base_models/models--stabilityai--stable-diffusion-3.5-medium/b940f670f0eda2d07fbb75229e779da1ad11eb80"
    config.pretrained.reward_model = 'pickscore'
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt_before_pruning = 8
    config.sample.skip_timesteps = [5,7]
    config.sample.lefts = [7,4]
    config.sample.num_image_per_prompt = config.sample.lefts[-1]
    # 所有进程一次batch总共算了多少不同的prompt
    config.sample.unique_prompts = int(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning)
    # 加一个sample_batches_per_epoch，用于记录每次采样多少个batch，因为每次采样多少个batch是根据num_image_per_prompt_before_pruning计算的，所以需要一个变量来记录
    config.sample.sample_batches_per_epoch = int(16/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning))
    config.sample.num_batches_per_epoch = int(16/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    # 新增：生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 更新：将时间戳附加到 save_dir
    config.run_name = f'pruning-8-to-4'
    config.save_dir = f'logs/pickscore/{config.run_name}_{timestamp}'
    # config.save_dir = 'logs/pickscore/sd3.5-M'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def geneval_sd3_pruning_2gpu():
    gpu_number = 2
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "/data11/xinyue.liu/sjy/TempFlow-GRPO/base_models/models--stabilityai--stable-diffusion-3.5-medium/b940f670f0eda2d07fbb75229e779da1ad11eb80"
    config.pretrained.reward_model = 'geneval'
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt_before_pruning = 16
    config.sample.skip_timesteps = [5,7]
    config.sample.lefts = [11,4]
    config.sample.num_image_per_prompt = config.sample.lefts[-1]
    # 所有进程一次batch总共算了多少不同的prompt
    config.sample.unique_prompts = int(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning)
    # 加一个sample_batches_per_epoch，用于记录每次采样多少个batch，因为每次采样多少个batch是根据num_image_per_prompt_before_pruning计算的，所以需要一个变量来记录
    config.sample.sample_batches_per_epoch = int(8/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning))
    config.sample.num_batches_per_epoch = int(8/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.04
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    # 新增：生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 更新：将时间戳附加到 save_dir
    config.run_name = f'geneval_pruning-16-to-4'
    config.save_dir = f'logs/geneval/{config.run_name}_{timestamp}'
    # config.save_dir = f'logs/geneval/sd3.5-M'
    config.reward_fn = {
        "geneval": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config

def geneval_sd3_pruning_4gpu():
    gpu_number = 4
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "/data11/xinyue.liu/sjy/TempFlow-GRPO/base_models/models--stabilityai--stable-diffusion-3.5-medium/b940f670f0eda2d07fbb75229e779da1ad11eb80"
    config.pretrained.reward_model = 'geneval'
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt_before_pruning = 16
    config.sample.skip_timesteps = [5,7]
    config.sample.lefts = [11,4]
    config.sample.num_image_per_prompt = config.sample.lefts[-1]
    # 所有进程一次batch总共算了多少不同的prompt
    config.sample.unique_prompts = int(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning)
    # 加一个sample_batches_per_epoch，用于记录每次采样多少个batch，因为每次采样多少个batch是根据num_image_per_prompt_before_pruning计算的，所以需要一个变量来记录
    config.sample.sample_batches_per_epoch = int(16/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning))
    config.sample.num_batches_per_epoch = int(16/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.04
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    # 新增：生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 更新：将时间戳附加到 save_dir
    config.run_name = f'geneval_pruning-16-to-4'
    config.save_dir = f'logs/geneval/{config.run_name}_{timestamp}'
    # config.save_dir = f'logs/geneval/sd3.5-M'
    config.reward_fn = {
        "geneval": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config

def pickscore_sd3_pruning_1gpu():
    gpu_number = 1
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "/data11/xinyue.liu/sjy/TempFlow-GRPO/base_models/models--stabilityai--stable-diffusion-3.5-medium/b940f670f0eda2d07fbb75229e779da1ad11eb80"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt_before_pruning = 8
    config.sample.num_image_per_prompt = 4
    config.sample.unique_prompts = int(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning)
    # 加一个sample_batches_per_epoch，用于记录每次采样多少个batch，因为每次采样多少个batch是根据num_image_per_prompt_before_pruning计算的，所以需要一个变量来记录
    config.sample.sample_batches_per_epoch = int(8/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning))
    config.sample.num_batches_per_epoch = int(8/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    # 新增：生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 更新：将时间戳附加到 save_dir
    config.save_dir = f'logs/pickscore/sd3.5-M-pruning-8-to-4_{timestamp}'
    # config.save_dir = 'logs/pickscore/sd3.5-M'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def clipscore_sd3():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.02
    config.sample.global_std = True
    config.sample.same_latent = True
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/clipscore/sd3.5-M'
    config.reward_fn = {
        "clipscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def pickscore_sd3_fast():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.train_num_steps = 2
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    # 这里固定为1
    config.sample.train_batch_size = 2
    config.sample.num_image_per_prompt = 24
    config.sample.mini_num_image_per_prompt = 9
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.mini_num_image_per_prompt/config.sample.num_image_per_prompt))
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.mini_num_image_per_prompt
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 1e-5
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.noise_level = 0.8
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    # 新增：生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 更新：将时间戳附加到 save_dir
    config.save_dir = f'logs/pickscore/sd3.5-M-fast-24_{timestamp}'
    # config.save_dir = 'logs/pickscore/sd3.5-M-fast'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config




def general_ocr_sd3_4gpu():
    gpu_number = 4
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(16/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/sd3.5-M'
    config.reward_fn = {
        "ocr": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def pickscore_sd3_4gpu():
    gpu_number= 4
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "/data11/xinyue.liu/sjy/TempFlow-GRPO/base_models/models--stabilityai--stable-diffusion-3.5-medium/b940f670f0eda2d07fbb75229e779da1ad11eb80"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt =16
    config.sample.num_batches_per_epoch = int(16/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    # 新增：生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 更新：将时间戳附加到 save_dir
    config.save_dir = f'logs/pickscore/sd3.5-M-4gpu-32-to-16_{timestamp}'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def pickscore_sd3_4gpu_pruning():
    gpu_number=4
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "/data11/xinyue.liu/sjy/TempFlow-GRPO/base_models/models--stabilityai--stable-diffusion-3.5-medium/b940f670f0eda2d07fbb75229e779da1ad11eb80"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt =16
    config.sample.num_image_per_prompt_before_pruning = 32
    config.sample.sample_batches_per_epoch = int(16/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt_before_pruning))
    config.sample.num_batches_per_epoch = int(16/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    # 新增：生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 更新：将时间戳附加到 save_dir
    config.save_dir = f'logs/pickscore/sd3.5-M-4gpu-32-to-16_{timestamp}'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def pickscore_sd3_8gpu():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "/data11/xinyue.liu/sjy/TempFlow-GRPO/base_models/models--stabilityai--stable-diffusion-3.5-medium/b940f670f0eda2d07fbb75229e779da1ad11eb80"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt =32
    config.sample.num_batches_per_epoch = int(16/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    # 新增：生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 更新：将时间戳附加到 save_dir
    config.save_dir = f'logs/pickscore/sd3.5-M-8gpu-32-to-16_{timestamp}'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def pickscore_sd3_24gpu():
    gpu_number=24
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "/data11/xinyue.liu/sjy/TempFlow-GRPO/base_models/models--stabilityai--stable-diffusion-3.5-medium/b940f670f0eda2d07fbb75229e779da1ad11eb80"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 48
    config.sample.num_batches_per_epoch = int(16/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    # 新增：生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 更新：将时间戳附加到 save_dir
    config.save_dir = f'logs/pickscore/sd3.5-M-24gpu-48-to-24_{timestamp}'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def general_ocr_sd3_1gpu():
    gpu_number = 1
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 8
    config.sample.num_batches_per_epoch = int(8/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/sd3.5-M'
    config.reward_fn = {
        "ocr": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def pickscore_flux():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # flux
    config.pretrained.model = "black-forest-labs/FLUX.1-dev"
    config.sample.num_steps = 6
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5

    config.resolution = 512
    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.sample.noise_level = 0.9
    config.mixed_precision = "bf16"
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = 'logs/pickscore/flux-group24'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def pickscore_flux_8gpu():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # flux
    config.pretrained.model = "black-forest-labs/FLUX.1-dev"
    config.sample.num_steps = 6
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5

    config.resolution = 512
    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.sample.noise_level = 0.9
    config.mixed_precision = "bf16"
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = 'logs/pickscore/flux-group24-8gpu'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def counting_flux_kontext():
    gpu_number=28
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/counting_edit")

    # sd3.5 medium
    config.pretrained.model = "black-forest-labs/FLUX.1-Kontext-dev"
    config.sample.num_steps = 6
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 2.5

    config.resolution = 512
    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 21
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 2 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.sample.noise_level = 0.9
    config.mixed_precision = "bf16"
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = 'logs/counting_edit/flux_kontext'
    config.reward_fn = {
        "image_similarity": 0.5,
        "geneval": 0.5,
    }
    config.per_prompt_stat_tracking = True
    return config

def pickscore_qwenimage():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # flux
    config.pretrained.model = "Qwen/Qwen-Image"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 4

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(32/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 4 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = False
    config.sample.noise_level = 1.2
    config.sample.sde_window_size = 2
    config.sample.sde_window_range = (0, config.sample.num_steps//2)
    config.mixed_precision = "bf16"
    config.use_lora = True
    config.activation_checkpointing = True
    config.fsdp_optimizer_offload = True
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = 'logs/pickscore/qwenimage'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def pickscore_qwenimage_8gpu():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # flux
    config.pretrained.model = "Qwen/Qwen-Image"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 4

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(32/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 4 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = False
    config.sample.noise_level = 1.2
    config.sample.sde_window_size = 2
    config.sample.sde_window_range = (0, config.sample.num_steps//2)
    config.mixed_precision = "bf16"
    config.use_lora = True
    config.activation_checkpointing = True
    config.fsdp_optimizer_offload = True
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = 'logs/pickscore/qwenimage'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def counting_qwenimage_edit():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/counting_edit")

    # flux
    config.pretrained.model = "Qwen/Qwen-Image-Edit"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 4

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(32/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 4 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = False
    config.sample.noise_level = 1.0
    config.sample.sde_window_size = 0
    # config.sample.sde_window_range = (0, config.sample.num_steps//2)
    config.mixed_precision = "bf16"
    config.use_lora = True
    config.activation_checkpointing = True
    config.fsdp_optimizer_offload = True
    config.save_freq = 60 # epoch
    config.eval_freq = 30
    config.save_dir = 'logs/pickscore/qwenimage_edit'
    config.reward_fn = {
        "image_similarity": 0.2,
        "geneval": 0.8,
    }
    config.per_prompt_stat_tracking = True
    return config

def counting_qwenimage_edit_fast():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/counting_edit")

    # flux
    config.pretrained.model = "Qwen/Qwen-Image-Edit"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 4

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(32/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 4 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = False
    config.sample.noise_level = 1.5
    config.sample.sde_window_size = 4
    config.sample.sde_window_range = (0, config.sample.num_steps//2)
    config.mixed_precision = "bf16"
    config.use_lora = True
    config.activation_checkpointing = True
    config.fsdp_optimizer_offload = True
    config.save_freq = 60 # epoch
    config.eval_freq = 30
    config.save_dir = 'logs/pickscore/qwenimage_edit'
    config.reward_fn = {
        "image_similarity": 0.2,
        "geneval": 0.8,
    }
    config.per_prompt_stat_tracking = True
    return config

def counting_qwenimage_edit_8gpu():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/counting_edit")

    # flux
    config.pretrained.model = "Qwen/Qwen-Image-Edit"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 4

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(32/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 4 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = False
    config.sample.noise_level = 1.0
    config.sample.sde_window_size = 0
    # config.sample.sde_window_range = (0, config.sample.num_steps//2)
    config.mixed_precision = "bf16"
    config.use_lora = True
    config.activation_checkpointing = True
    config.fsdp_optimizer_offload = True
    config.save_freq = 60 # epoch
    config.eval_freq = 30
    config.save_dir = 'logs/pickscore/qwenimage_edit'
    config.reward_fn = {
        "image_similarity": 0.2,
        "geneval": 0.8,
    }
    config.per_prompt_stat_tracking = True
    return config

def get_config(name):
    return globals()[name]()
