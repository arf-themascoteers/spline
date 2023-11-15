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
    return list(wavelengths_itr())


def get_wavelengths_str():
    return ",".join(get_wavelengths())


def get_rgb():
    return ["blue", "green", "red"]


if __name__ == "__main__":
    print(list(wavelengths_itr()))