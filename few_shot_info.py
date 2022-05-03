from myutils import Configs, auto_get_tag_names, get_datasets

if __name__ == "__main__":
    config = Configs.parse_from_argv()

    ds = get_datasets(f"{config.dataset_name}-base")['train']
    tag_names = auto_get_tag_names(config)
    tag_cnt = {}
    for tag in tag_names:
        if tag == 'O':
            continue
        tag_cnt[tag] = 0

    len_ds = len(ds)
    ds = ds.shuffle(config.few_shot_seed).select(list(range(int(len_ds * config.few_shot))))

    for tags in ds["tags"]:
        for tag in tags:
            if tag_names[tag] == 'O':
                continue
            tag_cnt[tag_names[tag]] += 1
    print(tag_cnt)
    entity_cnt = {}
    for tag in tag_names:
        if tag == 'O':
            continue
        entity_cnt[tag[2:]] = tag_cnt[f"B-{tag[2:]}"]
    print(entity_cnt)