import copy
import yaml


#把yaml的字典改为对象（递归实现），以便通过"对象.属性"的方式进行调用
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

#加载yaml文件为属性字典类（入口）
def load_yaml(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        raw_dict = yaml.safe_load(f)
    return AttrDict(raw_dict)


def merge_config(base, override):
    #使用版本修改内容递归覆盖原本内容，生成一个动态的版本配置实例
    #std版本可以直接读取config，无需走此路径
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


#生成版本完整配置的入口
def get_version_config(config):
    override = getattr(config, config.version, None)
    if isinstance(override, dict):
        return merge_config(config, override)
    return config


