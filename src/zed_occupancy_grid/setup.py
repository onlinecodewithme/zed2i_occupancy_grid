from setuptools import find_packages, setup

package_name = 'zed_occupancy_grid'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
         ['launch/zed_occupancy_grid.launch.py']),
        ('share/' + package_name + '/config',
         ['config/occupancy_grid.rviz']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='x4',
    maintainer_email='onlinecodewithme@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'zed_occupancy_grid_node = zed_occupancy_grid.zed_occupancy_grid_node:main',
            'zed_tf_setup = zed_occupancy_grid.tf_setup:main',
            'zed_loop_closure = zed_occupancy_grid.loop_closure:main'
        ],
    },
)
