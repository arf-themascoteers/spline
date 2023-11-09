def wavelengths_itr():
    wvs = []
    spec = 400
    while spec <= 2499.5:
        n_spec = spec
        if int(n_spec) == spec:
            n_spec = int(n_spec)
        wavelength = str(n_spec)
        yield wavelength
        spec = spec + 0.5
    return wvs


def get_wavelengths():
    return ['b1', 'blue', 'green', 'red', 'vnir1', 'vnir2',
     'vnir3', 'vnir4', 'vnir5', 'swir1', 'swir2', 'swir3', 'swir4']


def get_wavelengths_str():
    return ",".join(get_wavelengths())


def get_rgb():
    return ["blue", "green", "red"]