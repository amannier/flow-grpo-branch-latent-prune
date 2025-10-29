# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
# with the following modifications:
# - It uses the patched version of `sde_step_with_logprob` from `sd3_sde_with_logprob.py`.
# - It returns all the intermediate latents of the denoising process as well as the log probs of each denoising step.
from typing import Any, Dict, List, Optional, Union
import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from .sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from collections import defaultdict
import numpy as np
import traceback
from accelerate import Accelerator
import warnings
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

def make_debug_qkv_hook(layer_name, storage):  # 传 storage 避 global
    def debug_hook(module, input, output):
        # print(f"🚨 DEBUG: Hook 触发！层={layer_name}, 模块类型={type(module).__name__}, 输入形状={input[0].shape if input and len(input)>0 else 'None'}")  # 输入日志
        # print(f"🚨 DEBUG: 输出={output.shape if output is not None else 'None'}, 值示例={output[0,0,:3] if output is not None and output.numel()>0 else 'N/A'}")  # 值日志
        
        if output is not None and output.numel() > 0:
            try:
                storage[layer_name].append(output.detach().clone())
                # print(f"✅ APPEND 成功！storage[{layer_name}] 现在 len={len(storage[layer_name])}")
            except Exception as e:
                print(f"❌ APPEND 失败: {e}\n栈迹: {traceback.format_exc()}")
        else:
            print(f"⚠️ 输出无效！检查 forward 分支或形状 mismatch")
    
    return debug_hook

# 简化注册：单次，带 forward 包裹
def debug_register_and_run(transformer, latent_input, timestep, prompt_embeds, pooled_embeds, joint_kwargs):
    storage = defaultdict(list)
    handles = []
    
    # last_block = transformer.transformer_blocks[12]
    # cross_attn = last_block.attn
    # 用第一个transformer_block
    first_block = transformer.transformer_blocks[0]
    cross_attn = first_block.attn
    # print(f"🔍 确认路径: last_block={type(last_block).__name__}, cross_attn={type(cross_attn).__name__}")
    # print(f"🔍 确认路径: first_block={type(first_block).__name__}, cross_attn={type(cross_attn).__name__}")

    # # 假设cross_attn是注意力模块
    # print(f"注意力模块 heads: {cross_attn.heads if hasattr(cross_attn, 'heads') else '无heads属性，可能需检查config'}")
    # print(f"是否多头: {hasattr(cross_attn, 'heads') and cross_attn.heads > 1}")
        
    proj_map = {'Q': ('to_q', cross_attn.to_q), 'K': ('to_k', cross_attn.to_k), 'V': ('to_v', cross_attn.to_v)}
    
    for qkv_type, (name, module) in proj_map.items():
        hook_fn = make_debug_qkv_hook(name, storage)
        handle = module.register_forward_hook(hook_fn)
        handles.append(handle)
        # print(f"🎣 注册 {qkv_type} 到 {name} (类型: {type(module).__name__}, has base_layer={hasattr(module, 'base_layer')})")
    
    # 手动 forward（隔离 tqdm iter）
    # print("🔥 开始 forward 测试...")
    try:
        with torch.no_grad():
            noise_pred = transformer(
                hidden_states=latent_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                joint_attention_kwargs=joint_kwargs,
                return_dict=False,
            )[0]
        # print(f"✅ Forward 完成！noise_pred 形状={noise_pred.shape}")
    except Exception as e:
        print(f"❌ Forward 异常: {e}\n栈迹: {traceback.format_exc()}")
    
    # # 总结
    # print("\n🏆 DEBUG 总结:")
    # for key in ['to_q', 'to_k', 'to_v']:
    #     print(f"{key}: len(storage['{key}'])={len(storage[key])}")
    # if any(storage.values()):
    #     print("🎉 捕获示例: ", storage['to_q'][0].shape if storage['to_q'] else "N/A")
    #     print("🎉 捕获示例: ", storage['to_k'][0].shape if storage['to_k'] else "N/A")
    #     print("🎉 捕获示例: ", storage['to_v'][0].shape if storage['to_v'] else "N/A")
    
    # 清理
    for h in handles:
        h.remove()
    # print("🧹 Handles 已移除")

    # 计算attn_probs，没有attention_mask，不是时序相关的注意力 
    query = storage['to_q'][0]
    key = storage['to_k'][0]
    value = storage['to_v'][0]

    batch_size = query.shape[0]
    seq_len = query.shape[1]
    head_dim = query.shape[2] // cross_attn.heads
    scale = head_dim**-0.5

    dtype = query.dtype

    # 这里提升精度到float32，但似乎SD3.5M的attn没有这么干
    query = query.float()
    key = key.float()

    query = query.view(batch_size, -1, cross_attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, cross_attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, cross_attn.heads, head_dim).transpose(1, 2)

    query = query.reshape(batch_size * cross_attn.heads, seq_len, head_dim)
    key = key.reshape(batch_size * cross_attn.heads, seq_len, head_dim)
    value = value.reshape(batch_size * cross_attn.heads, seq_len, head_dim)

    baddbmm_input = torch.empty(
        query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
    )
    beta = 0

    attention_scores = torch.baddbmm(
        baddbmm_input,
        query,
        key.transpose(-1, -2),
        beta=beta,
        alpha=scale,
    )
    del baddbmm_input

    # 这里提升精度到float32，但似乎SD3.5M的attn没有这么干
    attention_scores = attention_scores.float()

    # attention_probs = attention_scores.softmax(dim=-1)
    attention_probs = attention_scores
    del attention_scores

    attention_probs = attention_probs.to(dtype) # [batch_size*num_heads, seq_q, seq_k]

    # Step 1: Reshape 到 [batch_size, num_heads, seq_q, seq_k]
    attention_probs_reshaped = attention_probs.view(
        batch_size, cross_attn.heads, seq_len, seq_len
    )  # [32, 24, 1024, 1024]

    # Step 2: 聚合 heads (mean over heads)
    attention_map_per_sample = attention_probs_reshaped.mean(dim=1)  # [32, 1024, 1024]

    # 可选：如果想 max 或其他
    # attention_map_per_sample = attention_probs_reshaped.max(dim=1).values  # [32, 1024, 1024]

    # 可选：CFG 切片 negative (前一半) 和 positive (后一半)
    negative_maps = attention_map_per_sample[:batch_size//2]  # [16, 1024, 1024]
    positive_maps = attention_map_per_sample[batch_size//2:]  # [16, 1024, 1024]

    return storage, positive_maps

def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

# 返回字典，prompt: [index_1, index_2, ...]
def find_prompt_positions_defaultdict(gathered_prompts):
    prompt_positions = defaultdict(list)

    for idx, prompt in enumerate(gathered_prompts):
        prompt_positions[prompt].append(idx)
    
    return dict(prompt_positions)

@torch.no_grad()
def pruning(latent, k):
    n = latent.shape[0]
    latent = latent.reshape(n, -1)
    if k >= n:
        return list(range(n))
    mu = latent.mean(dim=0, keepdim=True)        # (1,d)
    d2 = ((latent - mu) ** 2).sum(dim=1)         # (n,)
    centers = [int(torch.argmax(d2).item())]
    # 维护每个点到已选中心的最近距离
    min_d2 = ((latent - latent[centers[0]:centers[0]+1]) ** 2).sum(dim=1)
    for _ in range(1, k):
        nxt = int(torch.argmax(min_d2).item())
        centers.append(nxt)
        d2_new = ((latent - latent[nxt:nxt+1]) ** 2).sum(dim=1)
        min_d2 = torch.minimum(min_d2, d2_new)
    return centers

def heuristic_max_var_indices(nums, k):
    n = len(nums)
    if k >= n:
        return list(range(n)), np.var(nums)
    
    # 按值排序并保留原索引
    sorted_indices = np.argsort(nums)
    sorted_nums = nums[sorted_indices]
    
    best_var = -1
    best_indices = None
    
    # 枚举：最小取 i 个，最大取 k - i 个
    for i in range(1, k):
        if i > len(sorted_nums) or (k - i) > len(sorted_nums):
            continue  # 防御性检查
        
        # 选取两端
        selected_left = sorted_indices[:i]
        selected_right = sorted_indices[-(k - i):] if (k - i) > 0 else np.array([], dtype=int)
        selected_indices = np.concatenate([selected_left, selected_right])
        
        # 计算当前组合的方差
        subset = nums[selected_indices]
        var = np.var(subset)
        
        if var > best_var:
            best_var = var
            best_indices = selected_indices
    
    return list(best_indices), best_var

@torch.no_grad()
def mean_attn_max_var_pruning(attn_maps, k):
    attn_maps = attn_maps.cpu().numpy()
    n = attn_maps.shape[0]
    if k >= n:
        return list(range(n))
    means = np.mean(attn_maps, axis=(1, 2))
    max_var_mean, _ = heuristic_max_var_indices(means, k)
    return max_var_mean

@torch.no_grad()
def skip_pruning(self, temp_skip_schedular, scorer, prompt, noise_pred, latents, k, t):
    assert scorer is not None, "scorer 需要被定义来计算skip images的奖励"
    # print(temp_skip_schedular.timesteps, t)
    skip_latents, _, _, _ = sde_step_with_logprob(
        temp_skip_schedular, 
        noise_pred.float(), 
        t.unsqueeze(0), 
        latents.float(),
        noise_level=0,
    )

    skip_latents = (skip_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    skip_latents = skip_latents.to(dtype=self.vae.dtype)
    skip_image = self.vae.decode(skip_latents, return_dict=False)[0]
    skip_image = self.image_processor.postprocess(skip_image, output_type='pil') # =output_type  
    
    prompt_list = [prompt] * len(skip_image)
    skip_rewards = scorer(prompt_list, skip_image)
    skip_rewards = skip_rewards.cpu().numpy()

    # debug_folder = '/data11/xinyue.liu/sjy/flow_grpo_hcy/flow_grpo/debug_remainder'
    # print(prompt)
    # for score, img in zip(skip_rewards, skip_image):
    #     import os
    #     img_save_path = os.path.join(debug_folder, f'{t}_{score}.png')
    #     img.save(img_save_path)

    skip_max_var_indices, _ = heuristic_max_var_indices(skip_rewards, k)
    return skip_max_var_indices

@torch.no_grad()
def pipeline_with_logprob_hcy(
    self,
    scorer: Optional[torch.nn.Module] = None,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    prompts: Union[str, List[str]] = None,
    prompt_ids: Optional[torch.FloatTensor] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    text_encoders: Optional[List[Any]] = None,
    tokenizers: Optional[List[Any]] = None,
    num_images_per_prompt: Optional[int] = 1, # 这里运行起来竟然真是1，前文没有设置
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None, # torch.Size([config.sample.train_batch_size, 205, 4096])
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    skip_layer_guidance_scale: float = 2.8,
    accelerator: Optional[Accelerator] = None,
    # timestep_to_prune: Optional[int] = None,
    # num_to_delete_per_prompt: Optional[int] = None,
    unique_prompts_num: Optional[int] = None,
    epoch: Optional[int] = None,
    noise_level: float = 0.7,
    latent_extract_index: int = None,
    operate_diffsim_latent_index: Optional[List[int]] = None,
    skip_timesteps: Optional[List[int]] = None,
    lefts: Optional[list[int]] = None,
    skip_scheduler_list: Optional[list[FlowMatchEulerDiscreteScheduler]] = None,
):
    assert (skip_timesteps is None) == (lefts is None) == (skip_scheduler_list is None), \
        "skip_timesteps 与 lefts 与 skip_scheduler_list 必须同时提供或同时省略"

    if skip_timesteps is not None:  
        assert len(skip_timesteps) == len(lefts), \
            "skip_timesteps 与 lefts 与 skip_scheduler_list 的长度必须一致"

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._skip_layer_guidance_scale = skip_layer_guidance_scale
    self._clip_skip = clip_skip
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=self.clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents( # torch.Size([config.sample.train_batch_size, 16, 64, 64])
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    ).float()
    # print("latents shape:", latents.shape)
    # import sys; sys.exit()
    
    # 5. Prepare timesteps
    scheduler_kwargs = {}
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        **scheduler_kwargs,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # 6. Prepare image embeddings
    all_latents = [latents]
    all_log_probs = []

    attn_dict = defaultdict(list)

    # 7. Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        num_paded = 0
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])

            if operate_diffsim_latent_index is not None:
                if i in operate_diffsim_latent_index:
                    # print(f"🚨🚨🚨 DiffSim Hook at timestep {i}, t={t}")
                    debug_storage, attn_map = debug_register_and_run(
                        self.transformer, 
                        latent_model_input, 
                        timestep, 
                        prompt_embeds, 
                        pooled_prompt_embeds, 
                        self.joint_attention_kwargs
                    )
                    attn_dict[f'{i}'].append((debug_storage, attn_map))

            noise_pred = self.transformer( # torch.Size([config.sample.train_batch_size, 16, 64, 64])
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
            # noise_pred = noise_pred.to(prompt_embeds.dtype)

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                
            latents_dtype = latents.dtype
            latents_before = latents
            latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
                self.scheduler, 
                noise_pred.float(), 
                t.unsqueeze(0), 
                latents.float(),
                noise_level=noise_level,
            )
            
            all_latents.append(latents)
            all_log_probs.append(log_prob)
            # print(f"len(all_latents): {len(all_latents)}, all_latents[-1].shape:{all_latents[-1].shape}")
            # print(f"len(all_log_probs): {len(all_log_probs)}, all_log_probs[-1].shape:{all_log_probs[-1].shape}")

            # if latents.dtype != latents_dtype:
            #     latents = latents.to(latents_dtype)

            if latent_extract_index is not None and i == latent_extract_index:
                latent_extract = latents

            # if accelerator is not None and i == timestep_to_prune: # 后续调整
            if accelerator is not None and i in skip_timesteps:
                skip_t_index = skip_timesteps.index(i)
                # CFG只在上面的prompt_embeds添加了，所以后续重新编码prompt_embeds时候确实要手动添加neg_prompt_embed
                neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

                num_processes = accelerator.num_processes

                # left_total = latents.shape[0] * num_processes - unique_prompts_num * num_to_delete_per_prompt
                left_total = unique_prompts_num * lefts[skip_t_index]

                if accelerator.is_main_process:
                    if skip_t_index == len(skip_timesteps) - 1:
                        assert left_total % num_processes == 0, f"最后的 left_total={left_total} 不能被 num_processes={num_processes} 整除；会影响训练"
                        
                remainder = left_total % num_processes
                # if remainder != 0:
                #     print(
                #         f"left_total={left_total} 不能被 num_processes={num_processes} 整除；"
                #         f"无法平均分配（余数={remainder})。将进行不均匀分配。",
                #     )
                base_num_per_process = left_total // num_processes

                all_latents_tensor = torch.stack(all_latents, dim=1) # [latents.shape[0], len(all_latents), 16, 64, 64]
                all_log_probs_tensor = torch.stack(all_log_probs, dim=1) # [latents.shape[0], len(all_log_probs)]

                # 收集
                if num_paded != 0:
                    latents = accelerator.pad_across_processes([latents], dim=0)[0]
                    log_prob = accelerator.pad_across_processes([log_prob], dim=0)[0]
                    all_latents_tensor = accelerator.pad_across_processes([all_latents_tensor], dim=0)[0]
                    all_log_probs_tensor = accelerator.pad_across_processes([all_log_probs_tensor], dim=0)[0]
                    latents_before = accelerator.pad_across_processes([latents_before], dim=0)[0]
                    noise_pred = accelerator.pad_across_processes([noise_pred], dim=0)[0]
                    prompt_ids = accelerator.pad_across_processes([prompt_ids], dim=0)[0]

                latents_world = accelerator.gather(latents).to(accelerator.device) # 我认为是[num_processes * latents.shape[0], 16, 64, 64]
                log_probs_world = accelerator.gather(log_prob).to(accelerator.device) # [num_processes * latents.shape[0]]   
                all_latents_tensor_world = accelerator.gather(all_latents_tensor).to(accelerator.device) # [num_process * latents.shape[0], len(all_latents), 16, 64, 64]         
                all_log_probs_tensor_world = accelerator.gather(all_log_probs_tensor).to(accelerator.device) # [num_process * latents.shape[0], len(all_log_probs)]

                latents_before_world = accelerator.gather(latents_before).to(accelerator.device)
                noise_pred_world = accelerator.gather(noise_pred).to(accelerator.device)

                # accelerator gather不了prompts，所以gather prompt_ids然后解码
                prompt_ids_world = accelerator.gather(prompt_ids).to(accelerator.device)

                if num_paded !=0:
                    assert latents_world.shape[0] % num_processes == 0, '总gather数不能整除num_process，难道是pad没作用？'
                    block_size = latents_world.shape[0] // num_processes
                    valid_indices = []
                    for i in range(num_processes):
                        block_start = i * block_size
                        if i < num_processes - num_paded:
                            # Full block is valid
                            valid_indices.extend(range(block_start, block_start + block_size))
                        else:
                            # Exclude the last padded element
                            valid_indices.extend(range(block_start, block_start + block_size - 1))

                    valid_indices_tensor = torch.tensor(valid_indices, device=accelerator.device)

                    # Apply index_select to remove pads from each gathered tensor
                    latents_world = latents_world.index_select(dim=0, index=valid_indices_tensor)
                    log_probs_world = log_probs_world.index_select(dim=0, index=valid_indices_tensor)
                    all_latents_tensor_world = all_latents_tensor_world.index_select(dim=0, index=valid_indices_tensor)
                    all_log_probs_tensor_world = all_log_probs_tensor_world.index_select(dim=0, index=valid_indices_tensor)
                    latents_before_world = latents_before_world.index_select(dim=0, index=valid_indices_tensor)
                    noise_pred_world = noise_pred_world.index_select(dim=0, index=valid_indices_tensor)
                    prompt_ids_world = prompt_ids_world.index_select(dim=0, index=valid_indices_tensor)

                prompts_world = self.tokenizer.batch_decode(
                    prompt_ids_world, skip_special_tokens=True
                )

                prompt_positions = find_prompt_positions_defaultdict(prompts_world)

                left_latents_world = []
                left_log_probs_world = []
                left_all_latents_tensor_world = []
                left_all_log_probs_tensor_world = []
                left_prompts_world = []

                for prompt, positions in prompt_positions.items(): # 逐prompt剪枝
                    # left_per_prompt = len(positions) - num_to_delete_per_prompt
                    left_per_prompt = lefts[skip_t_index]
                    # assert left_per_prompt > 0, f"timestep{i}: images_per_prompt {len(positions)} must be larger than num_to_delete_per_prompt {num_to_delete_per_prompt}"

                    # p_latents = latents_world.index_select(0, torch.tensor(positions, device=accelerator.device))
                    p_latents_before = latents_before_world.index_select(0, torch.tensor(positions, device=accelerator.device))
                    p_noise_pred = noise_pred_world.index_select(0, torch.tensor(positions, device=accelerator.device))
                    # 筛选函数, 返回index_list
                    p_left_index = skip_pruning(self, skip_scheduler_list[skip_t_index], scorer, prompt, p_noise_pred, p_latents_before, left_per_prompt, t)

                    selected_index_world = torch.tensor([positions[index] for index in p_left_index], device=accelerator.device)

                    left_latents_world.append(latents_world.index_select(0, selected_index_world))
                    left_log_probs_world.append(log_probs_world.index_select(0, selected_index_world))
                    left_all_latents_tensor_world.append(all_latents_tensor_world.index_select(0, selected_index_world))
                    left_all_log_probs_tensor_world.append(all_log_probs_tensor_world.index_select(0, selected_index_world))
                    left_prompts_world.extend([prompt] * left_per_prompt)

                left_latents_world = torch.cat(left_latents_world, dim=0)
                left_log_probs_world = torch.cat(left_log_probs_world, dim=0)
                left_all_latents_tensor_world = torch.cat(left_all_latents_tensor_world, dim=0)
                left_all_log_probs_tensor_world = torch.cat(left_all_log_probs_tensor_world, dim=0)
                    
                # shuffle时保证所有进程的shuffle相同！
                g = torch.Generator(device=accelerator.device)
                g.manual_seed(42 + epoch)
                shuffle_idx = torch.randperm(left_total, generator=g, device=accelerator.device)

                left_latents_world = left_latents_world[shuffle_idx]
                left_log_probs_world = left_log_probs_world[shuffle_idx]
                left_all_latents_tensor_world = left_all_latents_tensor_world[shuffle_idx]
                left_all_log_probs_tensor_world = left_all_log_probs_tensor_world[shuffle_idx]
                left_prompts_world = [left_prompts_world[s.item()] for s in shuffle_idx]

                # 分配
                rank = accelerator.process_index
                num_elements_for_this_process = base_num_per_process + (1 if rank < remainder else 0)
                start_idx = sum(base_num_per_process + (1 if i < remainder else 0) for i in range(rank))
                # num_elements_for_this_process = base_num_per_process
                # start_idx = sum(base_num_per_process for _ in range(rank))
                end_idx = start_idx + num_elements_for_this_process
                latents = left_latents_world[start_idx: end_idx]
                log_prob = left_log_probs_world[start_idx: end_idx]
                all_latents_tensor = left_all_latents_tensor_world[start_idx: end_idx]
                all_latents = list(torch.unbind(all_latents_tensor, dim=1))
                all_log_probs_tensor = left_all_log_probs_tensor_world[start_idx: end_idx]
                all_log_probs = list(torch.unbind(all_log_probs_tensor, dim=1))
                prompts = left_prompts_world[start_idx: end_idx]

                if remainder != 0:
                    num_paded = num_processes - remainder
                else: 
                    num_paded = 0

                # 重编码prompt_ids和prompt_embeds，以及注意CFG！
                prompt_embeds, pooled_prompt_embeds = compute_text_embeddings( 
                    prompts, 
                    text_encoders, 
                    tokenizers, 
                    max_sequence_length=128, 
                    device=accelerator.device
                )
                prompt_ids = tokenizers[0](
                    prompts,
                    padding="max_length",
                    max_length=256,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(accelerator.device)

                sample_neg_prompt_embeds = neg_prompt_embed.repeat(num_elements_for_this_process, 1, 1)
                sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(num_elements_for_this_process, 1)

                if self.do_classifier_free_guidance:
                    prompt_embeds = torch.cat([sample_neg_prompt_embeds, prompt_embeds], dim=0)
                    pooled_prompt_embeds = torch.cat([sample_neg_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            
            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

    latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    latents = latents.to(dtype=self.vae.dtype)
    image = self.vae.decode(latents, return_dict=False)[0]
    image = self.image_processor.postprocess(image, output_type=output_type) # =output_type

    # Offload all models
    self.maybe_free_model_hooks()

    # if latent_extract_index is not None:
    #     return image, latent_extract, all_latents, all_log_probs, prompts
    # if operate_diffsim_latent_index is not None:
    #     return image, attn_dict, all_latents, all_log_probs, prompts
    return image, all_latents, all_log_probs, prompts
