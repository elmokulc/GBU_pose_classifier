# Filename: <setup.py>
# Copyright (C) <2021> Authors: <Pierre Vacher, Ludovic Charleux, Emile Roux, Christian Elmo Kulanesan>
# 
# This program is free software: you can redistribute it and / or 
# modify it under the terms of the GNU General Public License as published 
# by the Free Software Foundation, either version 2 of the License. 
# 
# This program is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of  
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the  
# GNU General Public License for more details.  
# 
# You should have received a copy of the GNU General Public License  
# along with this program.  If not, see <https://www.gnu.org/licenses/>. 
from setuptools import setup
import gbu

setup(name='gbu',
      version=gbu.__version__,
      description="Composite marker pose estimation using Python",
      long_description="Composite marker pose estimation using Python",
      author="Ludovic Charleux, Emile Roux, Christian Elmo, Pierre Vacher",
      author_email="ludovic.charleux@univ-smb.fr",
      license="GPL v3",
      packages=["gbu"],
      zip_safe=False,
      include_package_data=True,
      url="https://gitlab.com/symmehub/gbu",
      install_requires=[
          "numpy",
          "scipy",
          "pandas",
          # "opencv",
      ],
      )
