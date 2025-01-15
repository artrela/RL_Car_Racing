# Part 1: Setup Gymnasium & Car Racing Environment

## What is Gymnasium?
Gymnasium was originally created by [OpenAI](https://openai.com/index/openai-gym-beta/) as a method for standardizing Reinforcement Learning research. Their hope was to provide researchers and hobbyists a reliable platform to benchmark their achievements, thereby advancing the field. It allows for focus to remain on the algorithms themselves, rather than having to reproduce environments from scratch, speeding up development and equalizing comparison between the end product. 

More recently, in 2022, [Gym became Gymnasium](https://farama.org/Announcing-The-Farama-Foundation) after OpenAI moved away from maintenance of the core libraries. The **Farama Foundation** now maintains Gymnasium, as well as many other RL based training libaries, which you can explore [here](https://farama.org/projects). One of particular interest to MRSD robotics students may be [Gymnasium-Robotics](https://robotics.farama.org/), which uses the same Gymnsaium interface but with popular physics engines like Mujoco. 

## Starting with Gymnasium
If you have not used Gymnasium before, or would like a more involved explanation of its use case and history, please refer [here](www.fillintheblank.com). For these assignments, we will be using the [Box2D Car Racing Environment](https://gymnasium.farama.org/environments/box2d/car_racing/). A sample for what this environment looks like can be found below, along with installation instructions.


<p align="center">
    <img src=https://gymnasium.farama.org/_images/car_racing.gif alt=car-racing width=400 height=300/> 
</p>


```bash
pip install swig
pip install gymnasium[box2d]
pytest -m gym_install # test if install was successful
```

Before moving forward, it is crucial that ones understands the Gymnasium API. It fairly straightforward, a great starting place is to reference the [Gymnasium website](https://gymnasium.farama.org/), at minumum the `Basic Usage` and `Training an Agent` sections. Additionally, reference the function in `tests/test_generic.py`, named `test_gym_install()` to get a feel for the interface. 


```
@misc{towers2024gymnasiumstandardinterfacereinforcement,
      title={Gymnasium: A Standard Interface for Reinforcement Learning Environments}, 
      author={Mark Towers and Ariel Kwiatkowski and Jordan Terry and John U. Balis and Gianluca De Cola and Tristan Deleu and Manuel Goulão and Andreas Kallinteris and Markus Krimmel and Arjun KG and Rodrigo Perez-Vicente and Andrea Pierré and Sander Schulhoff and Jun Jet Tai and Hannah Tan and Omar G. Younis},
      year={2024},
      eprint={2407.17032},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.17032}, 
}
@misc{brockman2016openaigym,
      title={OpenAI Gym}, 
      author={Greg Brockman and Vicki Cheung and Ludwig Pettersson and Jonas Schneider and John Schulman and Jie Tang and Wojciech Zaremba},
      year={2016},
      eprint={1606.01540},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1606.01540}, 
}
```