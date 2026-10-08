This is the data set and software accompanying the paper "Faster Planar Graph Algorithms for Connectivity Problems via Meanders". It contains the C++ software that computes the upper bounds for the acyclic meandric systems (and also meandric numbers) and forest meandric systems. The software was developed using ChatGPT-5.6 Sol.

Contents:
- meander_upper.cpp:      the C++ program computing the acyclic meandric system upper bounds.
- meander-output.txt:     the documented output of the program for cross-validation.
- meander-numbers.txt:    the upper bounds for the acyclic meandric systems and meandric numbers.
- forest_upper.cpp:       the C++ program computing the forest meandric system upper bounds.
- forest-output.txt:      the documented output of the program for cross-validation.
- forest-numbers.txt:     the upper bounds for the forest meandric systems.

Main parameters:
--max-len:                the length of the standard representatives.
--primitive-len:          the length of the primitive jumps (default max-len).
--shift-jumps:            the number of jumps concatenated into one shift (default 50).