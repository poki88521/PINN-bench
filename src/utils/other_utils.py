

def get_test_data(example, data_config):
    if example.time_domain is None:
        x_test = example.geom.uniform_points(n=data_config.num_test, boundary=True)
    else:
        x_test = example.geomtime.uniform_points(n=data_config.num_test, boundary=True)
    if example.exact_func is None:
        u_true = example.load_data(x_test)
    else:
        u_true = example.u_exact_numpy(x_test)
    return x_test, u_true