dassl_src = [
    'dassl/dassl.h',
    'dassl/src/ddassl.f',
]

pside_src = [
    'pside/pside.h',
    'pside/src/pside.f',
]

py.extension_module(
    'dae4py',
    'dae4pymodule.c',
    dassl_src,
    pside_src,
    install: true,
    dependencies: [lapack_dep],
)
