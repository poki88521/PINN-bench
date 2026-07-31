import yaml


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)
            elif isinstance(value, list):
                # 如果列表内包含字典，也递归转换
                self[key] = [
                    AttrDict(item) if isinstance(item, dict) else item
                    for item in value
                ]

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

def load_yaml(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        raw_dict = yaml.safe_load(f)
    return AttrDict(raw_dict)