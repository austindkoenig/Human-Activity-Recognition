# <center>Human Activity Recognition</center>

<center>Austin Koenig</center>

## Introduction

### Data

The object of this project is to create a classifier which can differentiate between 14 different human activities from data collected via a triaxial accelerometer mounted to the subjects' wrists.

Following is the specifications of the device used to record the data:

#### Accelerometer Specifications

| Property | Description |
| --- | --- |
| Type | tri-axial accelerometer |
| Measurement range | [- 1.5g; + 1.5g] |
| Sensitivity | 6 bits per axis |
| Output data rate | 32 Hz |
| Location | attached to the right wrist of the user with:<br><ul><li>x axis: pointing toward the hand</li><li>y axis: pointing toward the left</li><li>z axis: perpendicular to the plane of the hand</li></ul> |

These data come from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer).

#### Brief Discussion on Possible Issues and Solutions

Consider the following table containing some statistics on the data.

| Activity Name | # Sequences | % Sequences | Min Seq Length | Max Seq Length |
| --- | --- | --- | --- | --- |
| Brush_teeth | 12 | 0.014303 | 844 | 3199 |
| Climb_stairs | 102 | 0.121573 | 166 | 3199 |
| Comb_hair | 31 | 0.036949 | 166 | 3199 |
| Descend_stairs | 42 | 0.05006 | 156 | 3199 |
| Drink_glass | 100 | 0.11919 | 156 | 3199 |
| Eat_meat | 5 | 0.005959 | 156 | 9318 |
| Eat_soup | 3 | 0.003576 | 156 | 9318 |
| Getup_bed | 101 | 0.120381 | 156 | 9318 |
| Liedown_bed | 28 | 0.033373 | 156 | 9318 |
| Pour_water | 100 | 0.11919 | 156 | 9318 |
| Sitdown_chair | 100 | 0.11919 | 125 | 9318 |
| Standup_chair | 102 | 0.121573 | 125 | 9318 |
| Use_telephone | 13 | 0.015495 | 125 | 9318 |
| Walk | 100 | 0.11919 | 125 | 9318 | 

## Results

For results, please see `./report.pdf`.