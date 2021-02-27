# CenterNet-ReImplementation
CenterNet Reimplementation  with simple code writing style

Some technical details that I found out when implementing the code:

- The bias of last layer to get the center map (i.e. heatmap) is set to a value of -2.19 as it will help the model converge on center faster.

- The loss of size is big compare to the other two since width and height are not normalized (in my code), I wonder if in the paper they would use wh that are normalized or not?

Todo: Trying to factorize the code, make it simpler to read
