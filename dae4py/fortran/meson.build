dassl_src = [
    'dassl/dassl.h',
    'dassl/src/ddassl.f',
]

radau_src = [
    'radau/radau.h',
    'radau/src/radau5.f',
    'radau/src/radau.f',
    # 'radau/src/decsol.f',
    # 'radau/src/dc_decsol.f',
    # 'radau/src/lapack.f',
    # 'radau/src/lapackc.f',
    'radau/src/dc_lapack.f',
]

pside_src = [
    'pside/pside.h',
    'pside/src/pside.f',
]

py.extension_module(
    'fortran',
    'fortran_module.c',
    dassl_src,
    pside_src,
    radau_src,
    subdir: 'dae4py',
    install: true,
    dependencies: [lapack_dep],
)
