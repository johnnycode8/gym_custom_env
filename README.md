<a name="readme-top"></a>

<h2 align="center">Gymnasium Custom Reinforcement Learning Environments</h2>

Tutorials on how to create custom Gymnasium-compatible Reinforcement Learning environments using the [Gymnasium Library](https://gymnasium.farama.org/), formerly OpenAI’s Gym library. Each tutorial has a companion video explanation and code walkthrough from my YouTube channel [@johnnycode](https://www.youtube.com/@johnnycode). If the code and video helped you, please consider:  
<a href='https://www.buymeacoffee.com/johnnycode'><img src="https://cdn.buymeacoffee.com/buttons/default-blue.png" alt="Buy Me A Coffee" height="41" width="174"></a>

## Custom Gym Environment part 1 - Warehouse Robot v0
This is a very basic tutorial showing end-to-end how to create a custom Gymnasium-compatible Reinforcement Learning environment. The tutorial is divided into three parts:
1. Model your problem.
2. Convert your problem into a Gymnasium-compatible environment.
3. Train your custom environment in two ways; using Q-Learning and using the Stable Baselines3 library. 

##### Code Reference:
* v0_warehouse_robot*.py

##### YouTube Tutorial:
<a href='https://youtu.be/AoGRjPt-vms&list=PL58zEckBH8fDt-F9LbpVASTor_jZzsxRg'><img src='https://img.youtube.com/vi/AoGRjPt-vms/0.jpg' width='400' alt='Build Custom Gymnasium Env'/></a>


## Custom Gym Environment part 2 - Visualization with Pygame
In part 1, we created a very simple custom Reinforcement Learning environment that is compatible with Farama Gymnasium (formerly OpenAI Gym). In this tutorial, we'll do a minor upgrade and visualize our environment using Pygame.

##### Code Reference:
* v0_warehouse_robot*.py

##### YouTube Tutorial:
<a href='https://youtu.be/9t64PFO7hr0&list=PL58zEckBH8fDt-F9LbpVASTor_jZzsxRg'><img src='https://img.youtube.com/vi/9t64PFO7hr0/0.jpg' width='400' alt='Build Custom Gymnasium Env with Pygame'/></a>



# Additional Resources

## How gymnasium.spaces.Box Works
The Box space type is used in many Gymnasium environments and you'll likely need it for your custom environment. The Box action space can be used to validate agent actions or generate random actions. The Box observation space can be used to validate the environment's state. This video explains and demos how to create boxes of different sizes/shapes, lower (low) and upper (high) boundaries, and as int/float data types.

##### YouTube Tutorial:
<a href='https://youtu.be/hI7UDemXVsk&list=PL58zEckBH8fDt-F9LbpVASTor_jZzsxRg'><img src='https://img.youtube.com/vi/hI7UDemXVsk/0.jpg' width='400' alt='How gymnasium.spaces.Box Works'/></a>

## Reinforcement Learning Tutorials
For more Reinforcement Learning and Deep Reinforcement Learning tutorials, check out my
[Gym Solutions](https://github.com/johnnycode8/gym_solutions) repository.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
