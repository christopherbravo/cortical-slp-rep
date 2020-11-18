# cortical-slp-rep
Cortical RA25 adaptation of slp-rep

## How to run:
1. Clone this repository to your computer. See instructons on how to do this [here](https://docs.github.com/en/free-pro-team@latest/github/creating-cloning-and-archiving-repositories/cloning-a-repository).  
2. ```cd``` into the repository and run the command ```go build  && cortical-slp-rep```. When you first run this command, go should automatically download all the packages required to run the model onto your computer. Once the command runs through, a GUI window should open up where you can interact with the model/explore its architecture.  
3. Click the "Train" button on the top left. When you do this, the following will happen:  
  i) The model will learn the RA25 patterns to zero error.  
  ii) The model will switch to an epoch of structured sleep trials where inputs are clamped onto the model but no target output signal is provided. The model learns by contrasting the following two phases in each trial: a) A one quarter plus phase where the hidden layer produces its own output in response to the inputs provided to the network (25 cycles) and b) A three quarter minus phase with oscillating inhibition switched on which uncovers useful contrastive states to the plus phase (75 cycles).  
  iii) After the structured sleep epoch, the model performs one final test and the run is ended.
 
Note: this sequence can be circumvented by using the buttons on the top toolbar to run a structured sleep trial or epoch at any time. Additionally, see the ```TrainTrial``` function in the cortical-slp-rep.go file which is an ideal place to set another specific sequence.
  
  ## Controlling sleep parameters:
 A set of input fields are availble in the control panel on the left of the screen with which specific aspects of the structured sleep minus phase oscillations can be controlled. These include a) start and stop points for the oscillation in the 75 cycle minus phase of each trial and b) sinusoidal wave parameters such as the period of each oscillation, the midline value, the amplitude etc.
 
 
