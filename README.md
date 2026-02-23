Sabrina and Will's 273 Final Project for LLO Satellite Orbit Determination Stability Analysis

Miscellaneous notes:
- The regular .shb files that possess the True spherical harmonic covariance matrices are simply way too big to work with except for severely truncated degrees (the order/degree 50 file is pretty workable at around only 26 MB), but GSFC did something really cool where they have clones of the covariance matrix. To my understanding, each clone is basically a Monte Carlo sample of the actual full-size covariance matrix. Each MC sample is only about 44 MB, which is _extremely_ manageable compared to the 8 TB that the regular cov. matrix would've been. A good starting point is going to be somehow averaging all of the clones (there are 500 total) to attain a "truth" harmonics matrix that we can use for satellite propogation. From there, we can take that truth matrix and work backwards. Will's job right now is just figuring out how to parse all of the clones. 

Brief List of To-Do's:
- Get a simulator set up
- Figure out what we want to simulate and how
- Do it
- Finish project
- easy peasy
