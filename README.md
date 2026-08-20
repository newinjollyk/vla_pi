# VLA on Raspberry Pi

A Vision-Language-Action (VLA) robotics project focused on deploying and evaluating a lightweight VLA pipeline on a Raspberry Pi.

## Overview

This project explores the development and deployment of a VLA-based robotic manipulation system. The robot uses visual observations and learned policies to perform manipulation tasks such as approaching, grasping, and placing objects.

The project includes:

- VLA model training and experimentation
- Robot manipulation data processing
- Inference and evaluation scripts
- Raspberry Pi deployment
- Gripper and camera integration
- Model checkpoint management

## Project Structure

```text
vla_pi/
├── scripts/              # Inference and utility scripts
├── test_top/             # Test images
├── models/               # Model-related files
├── configs/              # Configuration files
├── outputs/              # Local training outputs (not tracked by Git)
├── README.md
└── .gitignore
