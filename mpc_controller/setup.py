from setuptools import find_packages, setup
import os, glob

package_name = 'mpc_controller'

def get_data_files():
    data_files = [
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (os.path.join('share', package_name), ['mpc_controller/acados_ocp.json']),
    ]
    for root, dirs, files in os.walk('acados_solver_src'):
        file_list = [os.path.join(root, f) for f in files]
        if file_list: 
            dest_dir = os.path.join('share', package_name, root)
            data_files.append((dest_dir, file_list))
            
    return data_files

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=get_data_files(),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='developer',
    maintainer_email='janis.marcin02@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mpc_controller = mpc_controller.mpc:main'
        ],
    },
)
