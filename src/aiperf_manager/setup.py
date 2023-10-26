import setuptools

setuptools.setup(
    name = 'aiperf-tool',
    version = '1.0.0',
    packages = setuptools.find_packages(exclude=['*test*']),

    python_requires = '>=3.5',
    install_requires = [
        'requests',
    ],

    author = 'thu-pacman',
    author_email = 'zhaijidong@tsinghua.edu.cn',
    description = 'AIPERF control',
    license = 'MIT',
    url = 'https://github.com/thu-pacman/AIPerf',
    entry_points = {
        'console_scripts' : [
            'aiperf = aiperf_cmd.aiperfctl:parse_args'
        ]
    }
)
