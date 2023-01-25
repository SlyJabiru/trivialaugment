def remove_prefix(text, prefix):
    return text[len(prefix):] if text.startswith(prefix) else text

def remove_postfix(text, postfix):
    return text[:-len(postfix)] if text.endswith(postfix) else text


conf_file_path = 'confs/wresnet28x10_cifar100_b128_maxlr.1_ta_only_hard_from_huge_nowarmup_200epochs.yaml'

gpu_start = 4
gpu_end = 7

seed_start = 0
seed_end = 3

seeds = list(range(seed_start, seed_end+1))
gpu_ids = list(range(gpu_start, gpu_end+1))

tag = remove_postfix(remove_prefix(conf_file_path, 'confs/'), '.yaml')

for (seed, gpu) in list(zip(seeds, gpu_ids)):
    new_seed = str(seed).zfill(3)
    new_tag = f'{tag}_{new_seed}try'
    new_tmux_session = new_tag.replace('.', '')

    commands = f"""
tmux new -s {new_tmux_session}
conda activate trivial_augment
CUDA_VISIBLE_DEVICES={gpu} python -m TrivialAugment.train -c {conf_file_path} \
    --seed {seed} \
    --dataroot /hdd/hdd4/lsj/trivial_augment \
    --tag {new_tag} 2>&1 | tee stdouts/{new_tag}.log
    """

    # print(new_tag)
    print(commands)
    print()





# 
# 
# 
