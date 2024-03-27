## How to install PyKDL
### orocos 
#### Note only python 3.10 works
https://github.com/orocos/orocos_kinematics_dynamics

Install orocos c++ 
https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/INSTALL.md

Install python binding (PyKDL)
https://github.com/orocos/orocos_kinematics_dynamics/blob/master/python_orocos_kdl/INSTALL.md

```bash
# If use conda python
cmake .. -DPYTHON_VERSION=3.10
make -j
cp PyKDL.so ~/anaconda3/envs/py310/lib/python3.10/site-packages/
```


### urdf_parser_py 
https://github.com/ros/urdf_parser_py 

## kdl_parser_py
https://github.com/ros/kdl_parser
```bash
cd kdl_parser_py
pip install -e .
```