# ergm
Python implementation of exponential random graph models

**DISCLAIMER:** This package is still under development and woefully incomplete. I'd say "use at your own risk" but for now, it's really more like, "don't use." You have been warned! 

---

An exponential random graph model (ergm) is a probability distribution over graphs specified by a set of
_sufficient statistics_ and and corresponding parameters.

The probability density function takes the form

<img src="https://render.githubusercontent.com/render/math?math=P(G \mid \theta) = \frac{1}{Z}\exp\left(\sum_a \theta_a k_a(G)\right)">

Here, the k_a are the sufficient statistics (e.g. counts of edges, triangles, etc.) and the theta_a are the parameters.

---

Eventually, this package will support sampling and estimation of parameters from data. 