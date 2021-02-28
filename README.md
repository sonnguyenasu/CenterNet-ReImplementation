# CenterNet-ReImplementation
CenterNet Reimplementation  with simple code writing style

- Backbone: Resnet50, feed down to c5 and back at c3 stride (stride = 4)
- Head: Center map + width-height map + offset map

Some technical details that I found out when implementing the code:

- Width height output is possitive so I pass it through a element-wise exponential function. (borrow idea from FCOS)

- The bias of last layer to get the center map (i.e. heatmap) is set to a value of -2.19 as it will help the model converge on center faster.

- The loss of size is big compare to the other two since width and height are not normalized (in my code), I wonder if in the paper they would use wh that are normalized or not?

Todo: Trying to factorize the code, make it simpler to read

