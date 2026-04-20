# FFT_Based_AI_Image_Detection
When an image is considered as a signal of two variables, the fourier transform of that 2D signal can be taken. That is the "Fourier Transform of the Image" that shows the presence of repeated patters due to diffusion models and lack of natural noise, that can be used to distinguish between AI generated images and real images. 

---

# Colab
https://colab.research.google.com/drive/1sRa5_36ksIv_h7RpJIC7UdUFKtJegzuZ?usp=sharing
---

# Example Run
<img width="1024" height="559" alt="8c4bd2ed-0cdd-479a-904d-9c0f2addb17f" src="https://github.com/user-attachments/assets/99a180f9-d479-44d3-8f91-25cad0ae3f32" />
<img width="2383" height="740" alt="fft_radial_power" src="https://github.com/user-attachments/assets/c3672c09-1d13-44c7-9711-5f977f873567" />
<img width="2685" height="760" alt="fft_ai_grid_detection" src="https://github.com/user-attachments/assets/ce6221a8-4436-424b-b210-7acde6bfe867" />
<img width="2122" height="683" alt="fft_overview" src="https://github.com/user-attachments/assets/5f5fc91b-e2f8-42e5-9bcb-4a50fb0691e1" />


# Mathematical Background

\documentclass[12pt, a4paper]{article}

% ── Packages ─────────────────────────────────────────────────
\usepackage{amsmath, amssymb, amsthm}
\usepackage{mathtools}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{parskip}
\usepackage{microtype}
\usepackage{enumitem}
\usepackage{tcolorbox}
\tcbuselibrary{theorems, skins, breakable}

\geometry{margin=2.5cm}
\hypersetup{colorlinks=true, linkcolor=blue!60!black, urlcolor=blue!60!black}

% ── Theorem environments ──────────────────────────────────────
\theoremstyle{definition}
\newtheorem{definition}{Definition}[section]
\newtheorem{property}{Property}[section]
\newtheorem{remark}{Remark}[section]

% ── Coloured boxes ────────────────────────────────────────────
\tcbset{
  formulabox/.style={
    enhanced, breakable,
    colback=blue!4!white, colframe=blue!50!black,
    fonttitle=\bfseries, title={#1},
    left=6pt, right=6pt, top=4pt, bottom=4pt,
  }
}

% ── Custom commands ───────────────────────────────────────────
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\abs}[1]{\left\lvert #1 \right\rvert}
\newcommand{\norm}[1]{\left\lVert #1 \right\rVert}
\newcommand{\floor}[1]{\left\lfloor #1 \right\rfloor}

% ─────────────────────────────────────────────────────────────
\title{\textbf{Mathematical Foundations for \\
       FFT-Based AI Image Detection}\\[0.5em]
       \large From the Discrete Fourier Transform to\\
       Polar Feature Extraction and Statistical Learning\\
       Charitha Ranasinghe\\
       20.04.2026}
\author{}
\date{}
% ─────────────────────────────────────────────────────────────

\begin{document}
\maketitle
\tableofcontents
\newpage

% ════════════════════════════════════════════════════════════
\section{The Discrete Fourier Transform}
% ════════════════════════════════════════════════════════════

\subsection{One-Dimensional DFT}

Let $\mathbf{x} = \bigl(x_0, x_1, \ldots, x_{N-1}\bigr) \in \C^N$ be a finite
sequence of length $N$.

\begin{tcolorbox}[formulabox={1-D DFT}]
\begin{equation}
  X[k] \;=\; \sum_{n=0}^{N-1} x[n]\, e^{-j\,2\pi k n / N},
  \qquad k = 0, 1, \ldots, N-1.
  \label{eq:dft1d}
\end{equation}
\end{tcolorbox}

The corresponding \emph{inverse} DFT is:
\begin{equation}
  x[n] \;=\; \frac{1}{N}\sum_{k=0}^{N-1} X[k]\, e^{+j\,2\pi k n / N}.
  \label{eq:idft1d}
\end{equation}

Here $j = \sqrt{-1}$, and $W_N = e^{-j2\pi/N}$ is the primitive $N$-th root of
unity.  The basis functions $\{e^{j2\pi kn/N}\}$ are orthogonal:
\begin{equation}
  \sum_{n=0}^{N-1} e^{j2\pi m n/N}\, e^{-j2\pi k n/N}
  \;=\; N\,\delta_{mk},
\end{equation}
where $\delta_{mk}$ is the Kronecker delta.

\subsection{Two-Dimensional DFT}

A grayscale image is a matrix $\mathbf{I} \in \R^{H \times W}$, where
$I[m,n]$ is the intensity at pixel $(m,n)$.  The \textbf{2-D DFT} decomposes
the image into a superposition of 2-D complex exponentials (spatial frequency
components):

\begin{tcolorbox}[formulabox={2-D DFT}]
\begin{equation}
  F[u,v] \;=\; \sum_{m=0}^{H-1}\sum_{n=0}^{W-1}
               I[m,n]\;
               e^{-j2\pi\!\left(\tfrac{mu}{H}+\tfrac{nv}{W}\right)},
  \label{eq:dft2d}
\end{equation}
\end{tcolorbox}
for spatial frequencies $(u,v) \in \{0,\ldots,H-1\}\times\{0,\ldots,W-1\}$.
The pair $(u,v)$ encodes cycles per image in the vertical and horizontal
directions respectively.

The inverse transform recovers the image:
\begin{equation}
  I[m,n] \;=\; \frac{1}{HW}\sum_{u=0}^{H-1}\sum_{v=0}^{W-1}
               F[u,v]\;
               e^{+j2\pi\!\left(\tfrac{mu}{H}+\tfrac{nv}{W}\right)}.
  \label{eq:idft2d}
\end{equation}

\subsection{Separability}

Because the exponent separates multiplicatively, the 2-D DFT is
\emph{separable}: it equals the 1-D DFT applied along rows, followed by the
1-D DFT applied along columns:
\begin{equation}
  F[u,v] \;=\;
  \underbrace{\sum_{m=0}^{H-1}
    e^{-j2\pi mu/H}
  \overbrace{
    \left(\sum_{n=0}^{W-1} I[m,n]\,e^{-j2\pi nv/W}\right)
  }^{\text{row DFT}}}_{\text{column DFT of row DFTs}}.
\end{equation}

This separability is precisely what the Fast Fourier Transform (FFT) algorithm
exploits to reduce complexity from $\mathcal{O}(N^2)$ to
$\mathcal{O}(N\log N)$.

% ════════════════════════════════════════════════════════════
\section{Frequency Shifting and the DC Component}
% ════════════════════════════════════════════════════════════

\subsection{Zero-Frequency (DC) Component}

The coefficient at $(u,v)=(0,0)$ is called the \emph{DC component}:
\begin{equation}
  F[0,0] \;=\; \sum_{m=0}^{H-1}\sum_{n=0}^{W-1} I[m,n]
           \;=\; H W\,\bar{I},
  \label{eq:dc}
\end{equation}
where $\bar{I}$ denotes the mean pixel intensity.  It carries the total
\emph{energy bias} of the image.

\subsection{FFT Shift — Centring the Spectrum}

The raw DFT output places the DC component at the corner $(0,0)$.  Applying
the frequency-shift theorem via multiplication by $(-1)^{m+n}$:
\begin{equation}
  I[m,n]\cdot(-1)^{m+n}
  \;\xleftrightarrow{\;\mathcal{F}_{2D}\;}\;
  F\!\left[u-\tfrac{H}{2},\; v-\tfrac{W}{2}\right],
  \label{eq:fftshift}
\end{equation}
translates the DC component to the geometric centre $\bigl(\tfrac{H}{2},
\tfrac{W}{2}\bigr)$, making the spectrum visually interpretable.  We denote
the shifted spectrum
\begin{equation}
  \hat{F}[u,v] \;=\; F\!\left[u-\tfrac{H}{2},\; v-\tfrac{W}{2}\right].
\end{equation}

% ════════════════════════════════════════════════════════════
\section{Magnitude Spectrum and Log Scaling}
% ════════════════════════════════════════════════════════════

Since $\hat{F}[u,v] \in \C$, we define the \textbf{magnitude spectrum}:
\begin{equation}
  \mathcal{M}[u,v] \;=\; \abs{\hat{F}[u,v]}
                   \;=\; \sqrt{\operatorname{Re}\bigl(\hat{F}[u,v]\bigr)^2
                               + \operatorname{Im}\bigl(\hat{F}[u,v]\bigr)^2}.
\end{equation}

The dynamic range of $\mathcal{M}$ spans many orders of magnitude (the DC
coefficient can be $10^6\times$ larger than high-frequency coefficients).
Visualization and feature extraction therefore use \textbf{log-compression}:

\begin{tcolorbox}[formulabox={Log Magnitude Spectrum}]
\begin{equation}
  \mathcal{L}[u,v] \;=\; 20\log\bigl(1 + \abs{\hat{F}[u,v]}\bigr)
                   \;\approx\; 20\log\abs{\hat{F}[u,v]},
  \label{eq:logmag}
\end{equation}
\end{tcolorbox}
where $\log$ denotes the natural logarithm and the $+1$ is a numerical
stabilizer that prevents $\log(0)$.  The factor 20 converts to a
decibel-like scale.

\begin{remark}
  The phase spectrum $\varphi[u,v] = \angle\,\hat{F}[u,v]
  = \arctan\!\bigl(\operatorname{Im}/\operatorname{Re}\bigr)$ encodes the
  spatial alignment of features but carries less discriminative information for
  the natural-vs-AI classification task.
\end{remark}

% ════════════════════════════════════════════════════════════
\section{Power Spectrum and Radial Analysis}
% ════════════════════════════════════════════════════════════

\subsection{Power Spectral Density}

The \textbf{power spectrum} (periodogram) is
\begin{equation}
  \mathcal{P}[u,v] \;=\; \abs{\hat{F}[u,v]}^2.
\end{equation}
By Parseval's theorem the total power in the spatial domain equals the total
power in the frequency domain:
\begin{equation}
  \sum_{m,n} \abs{I[m,n]}^2 \;=\; \frac{1}{HW}
  \sum_{u,v} \abs{F[u,v]}^2.
\end{equation}

\subsection{DC Energy Fraction}

\begin{tcolorbox}[formulabox={DC Energy Fraction}]
\begin{equation}
  \eta_{\mathrm{DC}} \;=\;
  \frac{\mathcal{P}\!\left[\tfrac{H}{2},\tfrac{W}{2}\right]}
       {\displaystyle\sum_{u=0}^{H-1}\sum_{v=0}^{W-1}\mathcal{P}[u,v]}.
  \label{eq:dcfrac}
\end{equation}
\end{tcolorbox}

Natural images obey a \emph{pink-noise} (``$1/f$'') power law, concentrating
most energy near DC.  AI-generated images distribute energy more uniformly
across frequencies, so $\eta_{\mathrm{DC}}$ is a primary discriminative
feature.

\subsection{Radial Power Profile}

Define polar coordinates centred at the DC position:
\begin{equation}
  r[m,n] \;=\; \sqrt{\!\left(n - \tfrac{W}{2}\right)^{\!2}
                     + \left(m - \tfrac{H}{2}\right)^{\!2}},
  \qquad
  \theta[m,n] \;=\; \arctan\!\left(\frac{m-\tfrac{H}{2}}{n-\tfrac{W}{2}}\right).
\end{equation}

The \textbf{radial (azimuthal) average} at radius $\rho$ is:
\begin{equation}
  \bar{\mathcal{P}}(\rho) \;=\;
  \frac{1}{\abs{\mathcal{A}_\rho}}
  \sum_{(m,n)\,:\,\floor{r[m,n]}=\rho} \mathcal{P}[m,n],
  \label{eq:radial}
\end{equation}
where $\mathcal{A}_\rho$ is the set of pixels at integer radius $\rho$.

For natural photographs the radial profile follows
$\bar{\mathcal{P}}(\rho) \propto \rho^{-\alpha}$ with $\alpha \approx 2$
(the classical $1/f^2$ spectral law of natural images).  AI-generated images
deviate from this monotone decay, exhibiting bumps or plateaus at specific
radii.

% ════════════════════════════════════════════════════════════
\section{Polar Frequency Feature Extraction}
% ════════════════════════════════════════════════════════════

\subsection{Polar Binning}

Let $R_{\max} = \min\!\bigl(\tfrac{H}{2}, \tfrac{W}{2}\bigr)$ be the
largest representable radius, $N_r$ the number of radial rings, and $N_s$ the
number of angular sectors.  Define:
\begin{align}
  \Delta r      &= \frac{R_{\max}}{N_r}, \\[4pt]
  \Delta\theta  &= \frac{2\pi}{N_s}.
\end{align}

Each \textbf{polar bin} is the set of spectrum pixels falling in ring $i$ and
sector $j$:
\begin{equation}
  \mathcal{B}_{ij} \;=\; \bigl\{(m,n) \;\big|\;
    i\,\Delta r \le r[m,n] < (i+1)\Delta r,\;
    j\,\Delta\theta \le \theta[m,n] < (j+1)\Delta\theta
  \bigr\}.
  \label{eq:bin}
\end{equation}

\begin{tcolorbox}[formulabox={Polar Bin Feature}]
\begin{equation}
  \phi_{ij} \;=\; \frac{1}{\abs{\mathcal{B}_{ij}}}
  \sum_{(m,n)\in\mathcal{B}_{ij}} \mathcal{L}[m,n],
  \qquad
  i \in \{0,\ldots,N_r-1\},\;\; j \in \{0,\ldots,N_s-1\}.
  \label{eq:polarbinfeat}
\end{equation}
\end{tcolorbox}

With $N_r = 32$ rings and $N_s = 16$ sectors this yields a
$\mathbf{512}$\textbf{-dimensional feature vector}
$\boldsymbol{\phi} \in \R^{N_r N_s}$.

\subsection{Scalar Summary Features}

Three additional statistics are appended:

\subsubsection*{DC Energy Fraction}
Defined in \eqref{eq:dcfrac}.

\subsubsection*{Spectral Flatness (Wiener Entropy)}
The \textbf{spectral flatness measure} quantifies how noise-like (flat) the
power spectrum is:
\begin{tcolorbox}[formulabox={Spectral Flatness}]
\begin{equation}
  \gamma \;=\;
  \frac{\displaystyle\exp\!\left(
        \frac{1}{HW}\sum_{u,v}\ln\!\bigl(\mathcal{P}[u,v]+\varepsilon\bigr)
        \right)}
       {\displaystyle\frac{1}{HW}\sum_{u,v}\mathcal{P}[u,v]},
  \label{eq:flatness}
\end{equation}
\end{tcolorbox}
where $\varepsilon > 0$ is a small constant for numerical stability.  Pure
white noise gives $\gamma = 1$; a tonal (single-frequency) signal gives
$\gamma \to 0$.  The numerator is the geometric mean and the denominator is
the arithmetic mean of the power spectrum.

\subsubsection*{High-to-Low Frequency Energy Ratio}

Partition frequency space at radius threshold $\rho_0 = 0.3\,R_{\max}$:
\begin{align}
  E_{\mathrm{LF}} &= \frac{1}{\abs{\Omega_{\mathrm{in}}}}\!\!
                     \sum_{(m,n)\in\Omega_{\mathrm{in}}}\!\!\mathcal{P}[m,n],
  \qquad
  \Omega_{\mathrm{in}}  = \{(m,n): r[m,n] < \rho_0\},\\[6pt]
  E_{\mathrm{HF}} &= \frac{1}{\abs{\Omega_{\mathrm{out}}}}\!\!
                     \sum_{(m,n)\in\Omega_{\mathrm{out}}}\!\!\mathcal{P}[m,n],
  \qquad
  \Omega_{\mathrm{out}} = \{(m,n): r[m,n] \ge \rho_0\}.
\end{align}
\begin{equation}
  \lambda \;=\; \frac{E_{\mathrm{HF}}}{E_{\mathrm{LF}}}.
  \label{eq:hflf}
\end{equation}

The complete feature vector is therefore:
\begin{equation}
  \mathbf{x} \;=\; \bigl(\phi_{00},\;\ldots,\;\phi_{(N_r-1)(N_s-1)},\;
                         \eta_{\mathrm{DC}},\;\gamma,\;\lambda\bigr)
              \;\in\; \R^{N_r N_s + 3}.
  \label{eq:featurevec}
\end{equation}
With $N_r=32, N_s=16$ this gives $\mathbf{x} \in \R^{515}$.

% ════════════════════════════════════════════════════════════
\section{Residual Peak Detection (Grid Artifact Test)}
% ════════════════════════════════════════════════════════════

To detect the periodic grid artifacts introduced by AI generators (caused by
convolutional strides and attention tile sizes), the DC lobe is first removed.
Define the DC-exclusion mask with radius $\rho_{\mathrm{dc}}$:
\begin{equation}
  \mathcal{L}_{\mathrm{res}}[m,n] \;=\;
  \begin{cases}
    0                  & \text{if } r[m,n] \le \rho_{\mathrm{dc}},\\
    \mathcal{L}[m,n]   & \text{otherwise.}
  \end{cases}
\end{equation}

Let $Q_\tau$ denote the $\tau$-th quantile of $\{\mathcal{L}_{\mathrm{res}}
[m,n] : \mathcal{L}_{\mathrm{res}}[m,n]>0\}$.  The \textbf{peak count} is:
\begin{equation}
  \mathcal{K} \;=\;
  \abs{\bigl\{(m,n) \;\big|\; \mathcal{L}_{\mathrm{res}}[m,n] > Q_{0.80}\bigr\}}.
  \label{eq:peakcount}
\end{equation}

Large $\mathcal{K}$ (empirically $> 0.01\,HW$) indicates scattered, periodic
high-energy peaks characteristic of AI generation.

% ════════════════════════════════════════════════════════════
\section{Feature Standardization}
% ════════════════════════════════════════════════════════════

Before training, each feature dimension $k$ is \textbf{z-score standardized}
over the training set $\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^n$:
\begin{equation}
  \tilde{x}_k^{(i)} \;=\; \frac{x_k^{(i)} - \mu_k}{\sigma_k},
  \qquad
  \mu_k = \frac{1}{n}\sum_{i=1}^n x_k^{(i)},
  \quad
  \sigma_k = \sqrt{\frac{1}{n}\sum_{i=1}^n \bigl(x_k^{(i)}-\mu_k\bigr)^2}.
  \label{eq:zscore}
\end{equation}

% ════════════════════════════════════════════════════════════
\section{Classification Models}
% ════════════════════════════════════════════════════════════

\subsection{Logistic Regression}

Given standardised features $\tilde{\mathbf{x}} \in \R^d$, the probability of
class $y=1$ (AI-generated) is:
\begin{equation}
  P(y=1\mid\tilde{\mathbf{x}}) \;=\; \sigma\!\left(\mathbf{w}^\top\tilde{\mathbf{x}} + b\right),
  \qquad
  \sigma(z) = \frac{1}{1+e^{-z}}.
  \label{eq:logreg}
\end{equation}

Parameters $(\mathbf{w},b)$ are found by minimizing the regularized
\emph{cross-entropy loss}:
\begin{equation}
  \mathcal{L}(\mathbf{w},b) \;=\;
  -\frac{1}{n}\sum_{i=1}^n
  \Bigl[y^{(i)}\log \hat{p}^{(i)} + (1-y^{(i)})\log(1-\hat{p}^{(i)})\Bigr]
  + \frac{\lambda}{2}\norm{\mathbf{w}}^2,
  \label{eq:crossentropy}
\end{equation}
where $\hat{p}^{(i)} = P(y=1\mid\tilde{\mathbf{x}}^{(i)})$ and $\lambda$
controls $L_2$ regularization.

\subsection{Random Forest}

A random forest is an ensemble of $T$ decision trees
$\{h_t(\tilde{\mathbf{x}})\}_{t=1}^T$, each trained on a
\textbf{bootstrap sample} $\mathcal{D}_t \sim \mathcal{D}$ and using a random
feature subset of size $\lfloor\sqrt{d}\rfloor$ at each split.  The
class-probability estimate is the average vote:
\begin{equation}
  \hat{P}(y=1\mid\tilde{\mathbf{x}}) \;=\; \frac{1}{T}\sum_{t=1}^T
  \mathbf{1}\!\bigl[h_t(\tilde{\mathbf{x}})=1\bigr].
  \label{eq:rf}
\end{equation}

Each tree is grown by recursively partitioning at the feature $k^*$ and
threshold $\tau^*$ that minimise the \textbf{Gini impurity}:
\begin{equation}
  \mathrm{Gini}(\mathcal{S}) \;=\; 1 - \sum_{c \in \{0,1\}} p_c^2,
  \qquad
  (k^*,\tau^*) \;=\; \argmin_{k,\tau}
  \frac{\abs{\mathcal{S}_L}}{\abs{\mathcal{S}}}\mathrm{Gini}(\mathcal{S}_L)
  + \frac{\abs{\mathcal{S}_R}}{\abs{\mathcal{S}}}\mathrm{Gini}(\mathcal{S}_R),
  \label{eq:gini}
\end{equation}
where $\mathcal{S}_L,\mathcal{S}_R$ are the left/right splits.

Feature importance of feature $k$ is the mean decrease in Gini impurity
across all trees:
\begin{equation}
  \mathrm{MDA}(k) \;=\; \frac{1}{T}\sum_{t=1}^T
  \sum_{\text{node } s\text{ splits on }k}
  \frac{\abs{\mathcal{S}_s}}{\abs{\mathcal{D}}}
  \Bigl[\mathrm{Gini}(\mathcal{S}_s)
        - \frac{\abs{\mathcal{S}_{s,L}}}{\abs{\mathcal{S}_s}}\mathrm{Gini}(\mathcal{S}_{s,L})
        - \frac{\abs{\mathcal{S}_{s,R}}}{\abs{\mathcal{S}_s}}\mathrm{Gini}(\mathcal{S}_{s,R})
  \Bigr].
  \label{eq:mda}
\end{equation}

\subsection{Gradient Boosting}

Gradient boosting constructs an additive model
$F_T(\tilde{\mathbf{x}}) = \sum_{t=1}^T \nu\, h_t(\tilde{\mathbf{x}})$,
where $\nu$ is the learning rate and each tree $h_t$ fits the
\emph{negative gradient} (pseudo-residuals) of the loss at the current
model $F_{t-1}$:
\begin{equation}
  r_i^{(t)} \;=\; -\left[\frac{\partial}{\partial F(\tilde{\mathbf{x}}^{(i)})}
  \mathcal{L}\!\bigl(y^{(i)}, F(\tilde{\mathbf{x}}^{(i)})\bigr)\right]_{F=F_{t-1}}.
  \label{eq:boosting}
\end{equation}

For the binary log-loss $\mathcal{L}(y, p) = -y\log p - (1-y)\log(1-p)$ the
pseudo-residuals reduce to
$r_i^{(t)} = y^{(i)} - \sigma\!\bigl(F_{t-1}(\tilde{\mathbf{x}}^{(i)})\bigr)$.

% ════════════════════════════════════════════════════════════
\section{Model Evaluation}
% ════════════════════════════════════════════════════════════

\subsection{ROC Curve and AUC}

For a classifier with score $s(\mathbf{x})$ and decision threshold $\delta$,
define the true and false positive rates:
\begin{equation}
  \mathrm{TPR}(\delta) = \frac{TP(\delta)}{TP(\delta)+FN(\delta)},
  \qquad
  \mathrm{FPR}(\delta) = \frac{FP(\delta)}{FP(\delta)+TN(\delta)}.
\end{equation}

The \textbf{Receiver Operating Characteristic (ROC) curve} traces
$\bigl(\mathrm{FPR}(\delta), \mathrm{TPR}(\delta)\bigr)$ for $\delta$ sweeping
over all score values.  The \textbf{Area Under the ROC Curve (AUC)} is:
\begin{equation}
  \mathrm{AUC} \;=\; \int_0^1 \mathrm{TPR}\!\bigl(\mathrm{FPR}^{-1}(t)\bigr)\,dt
              \;=\; P\bigl(s(\mathbf{x}^+) > s(\mathbf{x}^-)\bigr),
  \label{eq:auc}
\end{equation}
where $\mathbf{x}^+$ is a random AI sample and $\mathbf{x}^-$ a natural one.

\subsection{5-Fold Cross-Validation}

The training set is randomly partitioned into $K=5$ folds
$\mathcal{D}_1,\ldots,\mathcal{D}_5$.  For fold $k$ the model is trained on
$\mathcal{D} \setminus \mathcal{D}_k$ and evaluated on $\mathcal{D}_k$:
\begin{equation}
  \widehat{\mathrm{AUC}}_{\mathrm{CV}} \;=\;
  \frac{1}{K}\sum_{k=1}^K \mathrm{AUC}_k,
  \qquad
  \widehat{\sigma}_{\mathrm{CV}} \;=\;
  \sqrt{\frac{1}{K-1}\sum_{k=1}^K
  \bigl(\mathrm{AUC}_k - \widehat{\mathrm{AUC}}_{\mathrm{CV}}\bigr)^2}.
  \label{eq:cv}
\end{equation}
\newpage
% ════════════════════════════════════════════════════════════
\section{Summary of Feature Vector}
% ════════════════════════════════════════════════════════════

\begin{table}[h!]
\centering
\renewcommand{\arraystretch}{1.4}
\begin{tabular}{@{}llcl@{}}
\toprule
\textbf{Component} & \textbf{Symbol} & \textbf{Dim.} & \textbf{Equation} \\
\midrule
Polar bin log-power   & $\phi_{ij}$          & 512 & \eqref{eq:polarbinfeat} \\
DC energy fraction    & $\eta_{\mathrm{DC}}$ & 1   & \eqref{eq:dcfrac}       \\
Spectral flatness     & $\gamma$             & 1   & \eqref{eq:flatness}     \\
HF / LF energy ratio  & $\lambda$            & 1   & \eqref{eq:hflf}         \\
\midrule
\textbf{Total}        &                      & \textbf{515} & \eqref{eq:featurevec} \\
\bottomrule
\end{tabular}
\caption{Components of the FFT feature vector $\mathbf{x} \in \R^{515}$.}
\end{table}

\end{document}
