#!/usr/bin/env python3
import numpy as np


class Layer_Input:
    # Forward pass
    def forward(self, inputs, training):
        self.output = inputs
