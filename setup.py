#!usr/bin/env python
# coding: utf-8
from setuptools import setup, find_packages
from subprocess import getoutput, call
# from setuptools import setup
# from setuptools.command.develop import develop
from setuptools.command.install import install
import tempfile

# class PostInstallCommand(install):
#     """Post-installation for installation mode."""
#     def run(self):
#         tmpdir=tempfile.mkdtemp()

#         d=getoutput("pip show numpy | grep Location | cut -d':' -f2")[1:]
#         call(f"git clone http://github.com/zygmuntz/hyperband {tmpdir}/hyperband".split())
#         call(f"mv {tmpdir}/hyperband {d}/".split())
#         call(f"pip download git+http://github.com/thuijskens/scikit-hyperband -d {tmpdir} --no-deps".split())
#         getoutput(f"cd {tmpdir};unzip scikit-hyperband-0.0.1.zip")
#         call(f"pip install -r {tmpdir}/scikit-hyperband/requirements.txt".split())
#         getoutput(f"rm -rf {tmpdir}/*")
#         call(f'pip install git+http://github.com/thuijskens/scikit-hyperband --target {tmpdir} --no-deps'.split())
#         call(f"mv {tmpdir}/hyperband {d}/scikit_hyperband".split())
#         install.run(self)
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION

setup(
        name             = "cvopt-study",
        version          = "0.4.2",
        description      = "Parameter search and feature selection's class, Integrated visualization and archive log.",
        license          = "BSD-2-Clause",
        author           = "gen/5",
        author_email     = "gen_fifth@outlook.jp",
        url              = "https://github.com/genfifth/cvopt.git",
        packages         = find_packages(),
    #     cmdclass={
    #     'install': PostInstallCommand,
    # },
        install_requires = ["numpy>=1.14", 
                            "pandas>=0.22.0", 
                            "scikit-learn>=0.19.1", 
                            "hyperopt>=0.1", 
                            "networkx==1.11", 
                            "GPy>=1.9.2", 
                            "gpyopt>=1.2.1",
                            "tzlocal>=2.0.0",
                            "bokeh>=0.12.14",
                            #"scikit-hyperband @ git+git://github.com/thuijskens/scikit-hyperband#egg=hyperband"
                            ],
        ),

