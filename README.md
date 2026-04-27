# AE502 Project

This repository contains analysis code for Earth-Moon L2 halo orbit station-keeping studies. The main workflow generates comparison figures and CSV summary tables for Northern and Southern branches under CR3BP and BR4BP modeling assumptions.

## Main Script

- `run_analysis.py`: runs the end-to-end station-keeping analysis, exports figures to `analysis_figures/`, and writes tables to `analysis_tables/`.

## Outputs

- `analysis_figures/`: exported plots used for analysis and reporting.
- `analysis_tables/`: exported CSV tables summarizing trade-study and station-keeping results.

## Notes

- The analysis depends on the local `astrokit/` package included in this repository.
- Figures and tables are generated from the current script settings in `run_analysis.py`.

## Generative AI Disclosure

This repository has been edited with assistance from generative AI tools. AI assistance was used to help draft and revise code, documentation, and plotting/layout updates. The project author remains responsible for reviewing, validating, and approving all technical content, numerical results, figures, and written material.
