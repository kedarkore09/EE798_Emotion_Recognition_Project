> README: Eficient Speech Emotion Recognition Using Multi-Scale CNN and
> Attention
>
> Kedar Kore Roll No.- 210503

1\. Project Overview

This repository implements a multi-modal emotion recognition system
using audio and text data from the IEMOCAP dataset. The model leverages
a combination of deep learning techniques, including MSCNN, statistical
pooling unit (SPU), and attention mechanisms, along with GloVe
embeddings for text representation.

The repository is organized into two phases:

> • Phase 1: Initial implementation of the emotion recognition model.
>
> • Phase 2: Enhanced implementation with Monte Carlo Dropout for
> uncertainty estimation.

2\. Dependencies

To install the necessary dependencies, run:

pip install -r requirements.txt

3\. Repository Structure

Common Files:

> • requirements.txt: List of dependencies.
>
> • README.md: Readme file, Project details.

Phase 1 Files:

> • train.py: Training script for the initial model.
>
> • test.py: Testing and evaluation script for the initial model.
>
> • analysis.py: Code for generating plots, classification reports, and
> confusion matrices.
>
> • EE798 implementation PDF.pdf: Phase 1 project report, results, and
> dataset description.

Phase 2 Files:

> • train part2.py: Training script for the enhanced model with Monte
> Carlo Dropout.
>
> • test part2.py: Testing the model with Monte Carlo Dropout.
>
> • evaluation part2.py: Code for generating Phase 2 evaluation metrics
> and plots.
>
> • EE798 projectsub2 pdf.pdf: Phase 2 project report, results, and
> dataset description.
>
> 1

4\. How to Run

Phase 1:

> 1\. Training the model:
>
> python train.py
>
> 2\. Testing:
>
> python test.py
>
> 3\. Evaluation:
>
> python analysis.py

Phase 2:

> 1\. Training the enhanced model with Monte Carlo Dropout:
>
> python train_part2.py
>
> 2\. Testing:
>
> python test_part2.py
>
> 3\. Evaluation:
>
> python evaluation_part2.py

5\. Results

Phase 1 Results: After running the Phase 1 scripts, you can visualize:

> • accuracy table.
>
> • Confusion matrix for the initial implementation.

Phase 2 Results: With Monte Carlo Dropout, the Phase 2 implementation
provides:

> • Enhanced uncertainty estimation.
>
> • Updated evaluation metrics.
>
> • confusion matrix visualizations.
>
> 2
