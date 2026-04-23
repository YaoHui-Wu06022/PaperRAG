# LONG SHORT-TERM MEMORY

Neural Computation 9(8):1735{1780, 1997

Sepp Hochreiter

Fakultat fur Informatik

Technische Universitat Munchen

80290 Munchen, Germany

hochreit@informatik.tu-muenchen.de

http://www7.informatik.tu-muenchen.de/~hochreit

Jurgen Schmidhuber

IDSIA

Corso Elvezia 36

6900 Lugano, Switzerland

juergen@idsia.ch

http://www.idsia.ch/~juergen

# Abst ract

Learning to store information over extended time intervals via recurrent backpropagation takes a very long time, mostly due to insucient, decaying error back ow. We briey review Hochreiter's 1991 analysis of this problem, then address it by introducing a novel, ecient, gradient-based method called \Long Short-Term Memory" (LSTM). Truncating the gradient where this does not do harm, LSTM can learn to bridge minimal time lags in excess of 1000 discrete time steps by enforcing constant error ow through \constant error carrousels" within special units. Multiplicative gate units learn to open and close access to the constant error ow. LSTM is local in space and time; its computational complexity per time step and weight is $O ( 1 )$ . Our experiments with articial data involve local, distributed, real-valued, and noisy pattern representations. In comparisons with RTRL, BPTT, Recurrent Cascade-Correlation, Elman nets, and Neural Sequence Chunking, LSTM leads to many more successful runs, and learns much faster. LSTM also solves complex, articial long time lag tasks that have never been solved by previous recurrent network algorithms.

# 1 INTRODUCTION

Recurrent networks can in principle use their feedback connections to store representations of recent input events in form of activations (\short-term memory", as opposed to \long-term memory" embodied by slowly changing weights). This is potentially signicant for many applications, including speech processing, non-Markovian control, and music composition (e.g., Mozer 1992). The most widely used algorithms for learning what to put in short-term memory, however, take too much time or do not work well at all, especially when minimal time lags between inputs and corresponding teacher signals are long. Although theoretically fascinating, existing methods do not provide clear practical advantages over, say, backprop in feedforward nets with limited time windows. This paper will review an analysis of the problem and suggest a remedy.

The problem. With conventional \Back-Propagation Through Time" (BPTT, e.g., Williams and Zipser 1992, Werbos 1988) or \Real-Time Recurrent Learning" (RTRL, e.g., Robinson and Fallside 1987), error signals \owing backwards in time" tend to either (1) blow up or (2) vanish: the temporal evolution of the backpropagated error exponentially depends on the size of the weights (Hochreiter 1991). Case (1) may lead to oscillating weights, while in case (2) learning to bridge long time lags takes a prohibitive amount of time, or does not work at all (see section 3).

The remedy. This paper presents \Long Short-Term Memory" (LSTM), a novel recurrent network architecture in conjunction with an appropriate gradient-based learning algorithm. LSTM is designed to overcome these error back-ow problems. It can learn to bridge time intervals in excess of 1000 steps even in case of noisy, incompressible input sequences, without loss of short time lag capabilities. This is achieved by an ecient, gradient-based algorithm for an architecture

enforcing constant (thus neither exploding nor vanishing) error ow through internal states of special units (provided the gradient computation is truncated at certain architecture-specic points | this does not aect long-term error ow though).

Outline of paper. Section 2 will briey review previous work. Section 3 begins with an outline of the detailed analysis of vanishing errors due to Hochreiter (1991). It will then introduce a naive approach to constant error backprop for didactic purposes, and highlight its problems concerning information storage and retrieval. These problems will lead to the LSTM architecture as described in Section 4. Section 5 will present numerous experiments and comparisons with competing methods. LSTM outperforms them, and also learns to solve complex, articial tasks no other recurrent net algorithm has solved. Section 6 will discuss LSTM's limitations and advantages. The appendix contains a detailed description of the algorithm (A.1), and explicit error ow formulae (A.2).

# 2 PREVIOUS WORK

This section will focus on recurrent nets with time-varying inputs (as opposed to nets with stationary inputs and fixpoint-based gradient calculations, e.g., Almeida 1987, Pineda 1987).

Gradient-descent variants. The approaches of Elman (1988), Fahlman (1991), Williams (1989), Schmidhuber (1992a), Pearlmutter (1989), and many of the related algorithms in Pearlmutter's comprehensive overview (1995) suer from the same problems as BPTT and RTRL (see Sections 1 and 3).

Time-delays. Other methods that seem practical for short time lags only are Time-Delay Neural Networks (Lang et al. 1990) and Plate's method (Plate 1993), which updates unit activations based on a weighted sum of old activations (see also de Vries and Principe 1991). Lin et al. (1995) propose variants of time-delay networks called NARX networks.

Time constants. To deal with long time lags, Mozer (1992) uses time constants inuencing changes of unit activations (deVries and Principe's above-mentioned approach (1991) may in fact be viewed as a mixture of TDNN and time constants). For long time lags, however, the time constants need external ne tuning (Mozer 1992). Sun et al.'s alternative approach (1993) updates the activation of a recurrent unit by adding the old activation and the (scaled) current net input. The net input, however, tends to perturb the stored information, which makes long-term storage impractical.

Ring's approach. Ring (1993) also proposed a method for bridging long time lags. Whenever a unit in his network receives conicting error signals, he adds a higher order unit inuencing appropriate connections. Although his approach can sometimes be extremely fast, to bridge a time lag involving 100 steps may require the addition of 100 units. Also, Ring's net does not generalize to unseen lag durations.

Bengio et al.'s approaches. Bengio et al. (1994) investigate methods such as simulated annealing, multi-grid random search, time-weighted pseudo-Newton optimization, and discrete error propagation. Their \latch" and \2-sequence" problems are very similar to problem 3a with minimal time lag 100 (see Experiment 3). Bengio and Frasconi (1994) also propose an EM approach for propagating targets. With n so-called \state networks", at a given time, their system can be in one of only $n$ dierent states. See also beginning of Section 5. But to solve continuous problems such as the \adding problem" (Section 5.4), their system would require an unacceptable number of states (i.e., state networks).

Kalman lters. Puskorius and Feldkamp (1994) use Kalman lter techniques to improve recurrent net performance. Since they use \a derivative discount factor imposed to decay exponentially the eects of past dynamic derivatives," there is no reason to believe that their Kalman Filter Trained Recurrent Networks will be useful for very long minimal time lags.

Second order nets. We will see that LSTM uses multiplicative units (MUs) to protect error ow from unwanted perturbations. It is not the rst recurrent net method using MUs though. For instance, Watrous and Kuhn (1992) use MUs in second order nets. Some differences to LSTM are: (1) Watrous and Kuhn's architecture does not enforce constant error ow and is not designed

to solve long time lag problems. (2) It has fully connected second-order sigma-pi units, while the LSTM architecture's MUs are used only to gate access to constant error ow. (3) Watrous and Kuhn's algorithm costs $O ( W ^ { 2 } )$ operations per time step, ours only $O ( W )$ , where $W$ is the number of weights. See also Miller and Giles (1993) for additional work on MUs.

Simple weight guessing. To avoid long time lag problems of gradient-based approaches we may simply randomly initialize all network weights until the resulting net happens to classify all training sequences correctly. In fact, recently we discovered (Schmidhuber and Hochreiter 1996, Hochreiter and Schmidhuber 1996, 1997) that simple weight guessing solves many of the problems in (Bengio 1994, Bengio and Frasconi 1994, Miller and Giles 1993, Lin et al. 1995) faster than the algorithms proposed therein. This does not mean that weight guessing is a good algorithm. It just means that the problems are very simple. More realistic tasks require either many free parameters (e.g., input weights) or high weight precision (e.g., for continuous-valued parameters), such that guessing becomes completely infeasible.

Adaptive sequence chunkers. Schmidhuber's hierarchical chunker systems (1992b, 1993) do have a capability to bridge arbitrary time lags, but only if there is local predictability across the subsequences causing the time lags (see also Mozer 1992). For instance, in his postdoctoral thesis (1993), Schmidhuber uses hierarchical recurrent nets to rapidly solve certain grammar learning tasks involving minimal time lags in excess of 1000 steps. The performance of chunker systems, however, deteriorates as the noise level increases and the input sequences become less compressible. LSTM does not suffer from this problem.

# 3 CONSTANT ERROR BACKPROP

# 3.1 EXPONENTIALLY DECAYING ERROR

Conventional BPTT (e.g. Williams and Zipser 1992). Output unit $k$ 's target at time $t$ is denoted by $d _ { k } ( t )$ . Using mean squared error, $k$ 's error signal is

$$
\vartheta_ {k} (t) = f _ {k} ^ {\prime} (n e t _ {k} (t)) (d _ {k} (t) - y ^ {k} (t)),
$$

where

$$
y ^ {i} (t) = f _ {i} (n e t _ {i} (t))
$$

is the activation of a non-input unit $i$ with dierentiable activation function $f _ { i }$ ,

$$
n e t _ {i} (t) = \sum_ {j} w _ {i j} y ^ {j} (t - 1)
$$

is unit $i$ 's current net input, and $w _ { i j }$ is the weight on the connection from unit $j$ to $i$ . Some non-output unit $j$ 's backpropagated error signal is

$$
\vartheta_ {j} (t) = f _ {j} ^ {\prime} (n e t _ {j} (t)) \sum_ {i} w _ {i j} \vartheta_ {i} (t + 1).
$$

The corresponding contribution to $w _ { j l }$ 's total weight update is $\alpha \vartheta _ { j } ( t ) y ^ { l } ( t - 1 )$ , where $\alpha$ is the learning rate, and $l$ stands for an arbitrary unit connected to unit $j$ .

Outline of Hochreiter's analysis (1991, page 19-21). Suppose we have a fully connected net whose non-input unit indices range from 1 to $n$ . Let us focus on local error ow from unit $u$ to unit $v$ (later we will see that the analysis immediately extends to global error ow). The error occurring at an arbitrary unit $u$ at time step $t$ is propagated \back into time" for $q$ time steps, to an arbitrary unit $v$ . This will scale the error by the following factor:

$$
\frac {\partial \vartheta_ {v} (t - q)}{\partial \vartheta_ {u} (t)} = \left\{ \begin{array}{c c} f _ {v} ^ {\prime} (n e t _ {v} (t - 1)) w _ {u v} & q = 1 \\ f _ {v} ^ {\prime} (n e t _ {v} (t - q)) \sum_ {l = 1} ^ {n} \frac {\partial \vartheta_ {l} (t - q + 1)}{\partial \vartheta_ {u} (t)} w _ {l v} & q > 1 \end{array} \right.. \tag {1}
$$

With $l _ { q } = v$ and $l _ { 0 } = u$ , we obtain:

$$
\frac {\partial \vartheta_ {v} (t - q)}{\partial \vartheta_ {u} (t)} = \sum_ {l _ {1} = 1} ^ {n} \dots \sum_ {l _ {q - 1} = 1} ^ {n} \prod_ {m = 1} ^ {q} f _ {l _ {m}} ^ {\prime} \left(n e t _ {l _ {m}} (t - m)\right) w _ {l _ {m} l _ {m - 1}} \tag {2}
$$

(proof by induction). The sum of the $n ^ { q - 1 }$ terms $\begin{array} { r } { \prod _ { m = 1 } ^ { q } f _ { l _ { m } } ^ { \prime } \big ( n e t _ { l _ { m } } \big ( t - m \big ) \big ) w _ { l _ { m } l _ { m - 1 } } } \end{array}$ determines the =1 m m m m 1 total error back ow (note that since the summation terms may have dierent signs, increasing the number of units $n$ does not necessarily increase error ow).

Intuitive explanation of equation (2). If

$$
| f _ {l _ {m}} ^ {\prime} (n e t _ {l _ {m}} (t - m)) w _ {l _ {m} l _ {m - 1}} | > 1. 0
$$

for all $m$ (as can happen, e.g., with linear $f _ { l _ { m } }$ ) then the largest product increases exponentially with $q$ . That is, the error blows up, and conicting error signals arriving at unit $v$ can lead to oscillating weights and unstable learning (for error blow-ups or bifurcations see also Pineda 1988, Baldi and Pineda 1991, Doya 1992). On the other hand, if

$$
| f _ {l _ {m} ^ {\prime}} (n e t _ {l _ {m}} (t - m)) w _ {l _ {m} l _ {m - 1}} | <   1. 0
$$

for all $m$ , then the largest product decreases exponentially with $q$ . That is, the error vanishes, and nothing can be learned in acceptable time.

If $f _ { l _ { m } }$ is the logistic sigmoid function, then the maximal value of $f _ { l _ { m } } ^ { \prime }$ is 0.25. If $y ^ { l _ { m - 1 } }$ is constant and not equal to zero, then $\left| f _ { l _ { m } } ^ { \prime } \left( n e t _ { l _ { m } } \right) w _ { l _ { m } l _ { m - 1 } } \right|$ m  takes on maximal values where

$$
w _ {l _ {m} l _ {m - 1}} = \frac {1}{y ^ {l _ {m - 1}}} \coth (\frac {1}{2} n e t _ {l _ {m}}),
$$

goes to zero for $| w _ { l _ { m } l _ { m - 1 } } | \to \infty$ , and is less than 1:0 for $| w _ { l _ { m } l _ { m - 1 } } | < 4 . 0$ (e.g., if the absolute maximal weight value $w _ { m a x }$ is smaller than 4.0). Hence with conventional logistic sigmoid activation functions, the error ow tends to vanish as long as the weights have absolute values below 4.0, especially in the beginning of the training phase. In general the use of larger initial weights will not help though | as seen above, for $| w _ { l _ { m } l _ { m - 1 } } | \to \infty$ the relevant derivative goes to zero \faster" than the absolute weight can grow (also, some weights will have to change their signs by crossing zero). Likewise, increasing the learning rate does not help either | it will not change the ratio of long-range error ow and short-range error ow. BPTT is too sensitive to recent distractions. (A very similar, more recent analysis was presented by Bengio et al. 1994).

Global error ow. The local error ow analysis above immediately shows that global error ow vanishes, too. To see this, compute

$$
\sum_ {u: u \text {o u t p u t u n i t}} \frac {\partial \vartheta_ {v} (t - q)}{\partial \vartheta_ {u} (t)}.
$$

Weak upper bound for scaling factor. The following, slightly extended vanishing error analysis also takes $n$ , the number of units, into account. For $q > 1$ , formula (2) can be rewritten as

$$
\left(W _ {u ^ {T}}\right) ^ {T} F ^ {\prime} (t - 1) \prod_ {m = 2} ^ {q - 1} \left(W F ^ {\prime} (t - m)\right) W _ {v} f _ {v} ^ {\prime} (n e t _ {v} (t - q)),
$$

where the weight matrix $W$ is dened by $[ W ] _ { i j } : = w _ { i j }$ , $v$ 's outgoing weight vector $W _ { v }$ is dened by $[ W _ { v } ] _ { i } : = [ W ] _ { i v } = w _ { i v }$ , $u$ 's incoming weight vector $W _ { u ^ { T } }$ is dened by $[ W _ { u ^ { T } } ] _ { i } : = [ W ] _ { u i } = w _ { u i }$ , and for $m = 1 , \ldots , q$ , $F ^ { \prime } ( t - m )$ is the diagonal matrix of rst order derivatives dened as: $[ F ^ { \prime } ( t { - } m ) ] _ { i j } : = 0$ if $i \neq j$ , and $[ F ^ { \prime } ( t - m ) ] _ { i j } : = f _ { i } ^ { \prime } ( n e t _ { i } ( t - m ) )$ otherwise. Here $T$ is the transposition operator, $[ A ] _ { i j }$ is the element in the $i$ -th column and $j$ -th row of matrix $A$ , and $[ x ] _ { i }$ is the $i$ -th component of vector $x$ .

Using a matrix norm $\| . \| _ { A }$ compatible with vector norm $\| . \| _ { x }$ , we dene

$$
f _ {m a x} ^ {\prime} := \max _ {m = 1, \dots , q} \{\parallel F ^ {\prime} (t - m) \parallel_ {A} \}.
$$

For $\operatorname* { m a x } _ { i = 1 , . . . , n } \{ | x _ { i } | \} \leq \| \ x \ \| _ { x }$ we get $\lvert x ^ { T } y \rvert \leq n \parallel x \parallel _ { x } \parallel _ { y } \parallel _ { x } .$ Since

$$
| f _ {v} ^ {\prime} (n e t _ {v} (t - q)) | \leq \| F ^ {\prime} (t - q) \| _ {A} \leq f _ {m a x} ^ {\prime},
$$

we obtain the following inequality:

$$
\mid \frac {\partial \vartheta_ {v} (t - q)}{\partial \vartheta_ {u} (t)} \mid \leq n (f _ {m a x} ^ {\prime}) ^ {q} \| W _ {v} \| _ {x} \| W _ {u ^ {T}} \| _ {x} \| W \| _ {A} ^ {q - 2} \leq n (f _ {m a x} ^ {\prime} \| W \| _ {A}) ^ {q}.
$$

This inequality results from

$$
\| W _ {v} \| _ {x} = \| W e _ {v} \| _ {x} \leq \| W \| _ {A} \| e _ {v} \| _ {x} \leq \| W \| _ {A}
$$

and

$$
\parallel W _ {u ^ {T}} \parallel_ {x} = \parallel e _ {u} W \parallel_ {x} \leq \parallel W \parallel_ {A} \parallel e _ {u} \parallel_ {x} \leq \parallel W \parallel_ {A},
$$

where $e _ { k }$ is the unit vector whose components are 0 except for the $k$ -th component, which is 1. Note that this is a weak, extreme case upper bound | it will be reached only if all $\parallel F ^ { \prime } ( t - m ) \parallel _ { A }$ take on maximal values, and if the contributions of all paths across which error fows back from unit $u$ to unit $v$ have the same sign. Large $\parallel \boldsymbol { W } \parallel _ { A }$ , however, typically result in small values of $\parallel F ^ { \prime } ( t - m ) \parallel _ { A }$ , as conrmed by experiments (see, e.g., Hochreiter 1991).

For example, with norms

$$
\parallel W \parallel_ {A} := \max _ {r} \sum_ {s} | w _ {r s} |
$$

and

$$
\parallel x \parallel_ {x} := \max _ {r} | x _ {r} |,
$$

we have $f _ { m a x } ^ { \prime } = 0 . 2 5$ for the logistic sigmoid. We observe that if

$$
| w _ {i j} | \leq w _ {m a x} <   \frac {4 . 0}{n} \forall i, j,
$$

then $\parallel W \parallel _ { A } \leq n w _ { m a x } < 4 . 0$ will result in exponential decay | by setting $\begin{array} { r } { \tau : = \left( \frac { n w _ { m a x } } { 4 . 0 } \right) < 1 . 0 } \end{array}$

$$
\mid \frac {\partial \vartheta_ {v} (t - q)}{\partial \vartheta_ {u} (t)} \mid \leq n (\tau) ^ {q}.
$$

We refer to Hochreiter's 1991 thesis for additional results.

# 3.2 CONSTANT ERROR FLOW: NAIVE APPROACH

A single unit. To avoid vanishing error signals, how can we achieve constant error ow through a single unit $j$ with a single connection to itself? According to the rules above, at time $t$ , $j$ 's local error back ow is $\vartheta _ { j } ( t ) = f _ { j } ^ { \prime } ( n e t _ { j } ( t ) ) \vartheta _ { j } ( t + 1 ) w _ { j j }$ . To enforce constant error ow through $j$ , we require

$$
f _ {j} ^ {\prime} (n e t _ {j} (t)) w _ {j j} = 1. 0.
$$

Note the similarity to Mozer's xed time constant system (1992) | a time constant of 1:0 is appropriate for potentially innite time lags1.

The constant error carrousel. Integrating the dierential equation above, we obtain $\begin{array} { r } { f _ { j } ( n e t _ { j } ( t ) ) = \frac { n e t _ { j } ( t ) } { w _ { j j } } } \end{array}$ for arbitrary $n e t _ { j } ( t )$ . This means: $f _ { j }$ has to be linear, and unit $j$ 's activation has to remain constant:

$$
y _ {j} (t + 1) = f _ {j} \left(n e t _ {j} (t + 1)\right) = f _ {j} \left(w _ {j j} y ^ {j} (t)\right) = y ^ {j} (t).
$$

In the experiments, this will be ensured by using the identity function $f _ { j } : f _ { j } ( x ) = x , \forall x$ , and by setting $w _ { j j } = 1 . 0$ . We refer to this as the constant error carrousel (CEC). CEC will be LSTM's central feature (see Section 4).

Of course unit $j$ will not only be connected to itself but also to other units. This invokes two obvious, related problems (also inherent in all other gradient-based approaches):

1. Input weight conict: for simplicity, let us focus on a single additional input weight $w _ { j i }$ . Assume that the total error can be reduced by switching on unit $j$ in response to a certain input, and keeping it active for a long time (until it helps to compute a desired output). Provided $i$ is nonzero, since the same incoming weight has to be used for both storing certain inputs and ignoring others, $w _ { j i }$ will often receive conicting weight update signals during this time (recall that $j$ is linear): these signals will attempt to make $w _ { j i }$ participate in (1) storing the input (by switching on $j$ ) and (2) protecting the input (by preventing $j$ from being switched o by irrelevant later inputs). This conict makes learning dicult, and calls for a more context-sensitive mechanism for controlling \write operations" through input weights.

2. Output weight conict: assume $j$ is switched on and currently stores some previous input. For simplicity, let us focus on a single additional outgoing weight $w _ { k j }$ . The same $w _ { k j }$ has to be used for both retrieving $j$ 's content at certain times and preventing $j$ from disturbing $k$ at other times. As long as unit $j$ is non-zero, $w _ { k j }$ will attract conicting weight update signals generated during sequence processing: these signals will attempt to make $w _ { k j }$ participate in (1) accessing the information stored in $j$ and | at dierent times | (2) protecting unit $k$ from being perturbed by $j$ . For instance, with many tasks there are certain \short time lag errors" that can be reduced in early training stages. However, at later training stages $j$ may suddenly start to cause avoidable errors in situations that already seemed under control by attempting to participate in reducing more dicult \long time lag errors". Again, this conict makes learning dicult, and calls for a more context-sensitive mechanism for controlling \read operations" through output weights.

Of course, input and output weight conicts are not specic for long time lags, but occur for short time lags as well. Their eects, however, become particularly pronounced in the long time lag case: as the time lag increases, (1) stored information must be protected against perturbation for longer and longer periods, and | especially in advanced stages of learning | (2) more and more already correct outputs also require protection against perturbation.

Due to the problems above the naive approach does not work well except in case of certain simple problems involving local input/output representations and non-repeating input patterns (see Hochreiter 1991 and Silva et al. 1996). The next section shows how to do it right.

# 4 LONG SHORT-TERM MEMORY

Memory cells and gate units. To construct an architecture that allows for constant error flow through special, self-connected units without the disadvantages of the naive approach, we extend the constant error carrousel CEC embodied by the self-connected, linear unit $j$ from Section 3.2 by introducing additional features. A multiplicative input gate unit is introduced to protect the memory contents stored in $j$ from perturbation by irrelevant inputs. Likewise, a multiplicative output gate unit is introduced which protects other units from perturbation by currently irrelevant memory contents stored in $j$ .

The resulting, more complex unit is called a memory cel l (see Figure 1). The $j$ -th memory cell is denoted $c _ { j }$ . Each memory cell is built around a central linear unit with a xed self-connection (the CEC). In addition to $n e t _ { c _ { j } }$ , $c _ { j }$ gets input from a multiplicative unit $o u t _ { j }$ (the \output gate"), and from another multiplicative unit $i n _ { j }$ (the \input gate"). $i n _ { j }$ 's activation at time $t$ is denoted by $y ^ { i n _ { j } } ( t )$ , $o u t _ { j }$ 's by $y ^ { o u t _ { j } } ( t )$ . We have

$$
y ^ {o u t _ {j}} (t) = f _ {o u t _ {j}} (n e t _ {o u t _ {j}} (t)); y ^ {i n _ {j}} (t) = f _ {i n _ {j}} (n e t _ {i n _ {j}} (t));
$$

where

$$
n e t _ {o u t j} (t) = \sum_ {u} w _ {o u t j u} y ^ {u} (t - 1),
$$

and

$$
n e t _ {i n _ {j}} (t) = \sum_ {u} w _ {i n _ {j} u} y ^ {u} (t - 1).
$$

We also have

$$
n e t _ {c _ {j}} (t) = \sum_ {u} w _ {c _ {j} u} y ^ {u} (t - 1).
$$

The summation indices $\boldsymbol { u }$ may stand for input units, gate units, memory cells, or even conventional hidden units if there are any (see also paragraph on \network topology" below). All these dierent types of units may convey useful information about the current state of the net. For instance, an input gate (output gate) may use inputs from other memory cells to decide whether to store (access) certain information in its memory cell. There even may be recurrent self-connections like $w _ { c _ { j } c _ { j } }$ . It is up to the user to dene the network topology. See Figure 2 for an example.

jAt time $t$ , $c _ { j }$ 's output $y ^ { c _ { j } } ( t )$ is computed as

$$
y ^ {c _ {j}} (t) = y ^ {o u t _ {j}} (t) h (s _ {c _ {j}} (t)),
$$

where the \internal state" $s _ { c _ { j } } ( t )$ is

$$
s _ {c _ {j}} (0) = 0, s _ {c _ {j}} (t) = s _ {c _ {j}} (t - 1) + y ^ {i n _ {j}} (t) g \left(n e t _ {c _ {j}} (t)\right) \mathrm {f o r} t > 0.
$$

The dierentiable function $g$ squashes $n e t _ { c _ { j } }$ ; the dierentiable function $h$ scales memory cell outputs computed from the internal state $s _ { c _ { j } }$ .

![](images/82c0ec97bdeee5472bb0e586a78715d7cc3f2b122f5c3077c4e9eb37961ec060.jpg)  
Figure 1: Architecture of memory cel l $c _ { j }$ (the box) and its gate units $i n _ { j } , o u t _ { j }$ . The self-recurrent connection (with weight 1.0) indicates feedback with a delay of 1 time step. It builds the basis of the \constant error carrousel" CEC. The gate units open and close access to CEC. See text and appendix A.1 for details.

Why gate units? To avoid input weight conicts, $i n _ { j }$ controls the error ow to memory cell $c _ { j }$ 's input connections $w _ { c _ { j } i }$ . To circumvent $c _ { j }$ 's output weight conicts, $o u t _ { j }$ controls the error ow from unit $j$ 's output connections. In other words, the net can use $i n _ { j }$ to decide when to keep or override information in memory cell $c _ { j }$ , and $o u t _ { j }$ to decide when to access memory cell $c _ { j }$ and when to prevent other units from being perturbed by $c _ { j }$ (see Figure 1).

Error signals trapped within a memory cell's CEC cannot change { but dierent error signals owing into the cell (at dierent times) via its output gate may get superimposed. The output gate will have to learn which errors to trap in its CEC, by appropriately scaling them. The input

gate will have to learn when to release errors, again by appropriately scaling them. Essentially, the multiplicative gate units open and close access to constant error ow through CEC.

Distributed output representations typically do require output gates. Not always are both gate types necessary, though | one may be sucient. For instance, in Experiments 2a and 2b in Section 5, it will be possible to use input gates only. In fact, output gates are not required in case of local output encoding | preventing memory cells from perturbing already learned outputs can be done by simply setting the corresponding weights to zero. Even in this case, however, output gates can be benecial: they prevent the net's attempts at storing long time lag memories (which are usually hard to learn) from perturbing activations representing easily learnable short time lag memories. (This will prove quite useful in Experiment 1, for instance.)

Network topology. We use networks with one input layer, one hidden layer, and one output layer. The (fully) self-connected hidden layer contains memory cells and corresponding gate units (for convenience, we refer to both memory cells and gate units as being located in the hidden layer). The hidden layer may also contain \conventional" hidden units providing inputs to gate units and memory cells. All units (except for gate units) in all layers have directed connections (serve as inputs) to all units in the layer above (or to all higher layers { Experiments 2a and 2b).

Memory cell blocks. $S$ memory cells sharing the same input gate and the same output gate form a structure called a \memory cell block of size S". Memory cell blocks facilitate information storage | as with conventional neural nets, it is not so easy to code a distributed input within a single cell. Since each memory cell block has as many gate units as a single memory cell (namely two), the block architecture can be even slightly more ecient (see paragraph \computational complexity"). A memory cell block of size 1 is just a simple memory cell. In the experiments (Section 5), we will use memory cell blocks of various sizes.

Learning. We use a variant of RTRL (e.g., Robinson and Fallside 1987) which properly takes into account the altered, multiplicative dynamics caused by input and output gates. However, to ensure non-decaying error backprop through internal states of memory cells, as with truncated BPTT (e.g., Williams and Peng 1990), errors arriving at \memory cell net inputs" (for cell $c _ { j }$ , this includes $n e t _ { c _ { j } }$ , $n e t _ { i n _ { j } }$ , $n e t _ { o u t _ { j } }$ ) do not get propagated back further in time (although they do serve to change the incoming weights). Only within2 memory cells, errors are propagated back through previous internal states $s _ { c _ { j } }$ . To visualize this: once an error signal arrives at a memory cell output, it gets scaled by output gate activation and $h ^ { \prime }$ . Then it is within the memory cell's CEC, where it can ow back indenitely without ever being scaled. Only when it leaves the memory cell through the input gate and $g$ , it is scaled once more by input gate activation and $g ^ { \prime }$ . It then serves to change the incoming weights before it is truncated (see appendix for explicit formulae).

Computational complexity. As with Mozer's focused recurrent backprop algorithm (Mozer 1989), only the derivatives $\frac { \partial s _ { c _ { j } } } { \partial w _ { i l } }$ @w @ sc need to be stored and updated. Hence the LSTM algorithm is very ecient, with an excellent update complexity of $O ( W )$ , where $W$ the number of weights (see details in appendix A.1). Hence, LSTM and BPTT for fully recurrent nets have the same update complexity per time step (while RTRL's is much worse). Unlike full BPTT, however, LSTM is local in space and time3: there is no need to store activation values observed during sequence processing in a stack with potentially unlimited size.

Abuse problem and solutions. In the beginning of the learning phase, error reduction may be possible without storing information over time. The network will thus tend to abuse memory cells, e.g., as bias cells (i.e., it might make their activations constant and use the outgoing connections as adaptive thresholds for other units). The potential diculty is: it may take a long time to release abused memory cells and make them available for further learning. A similar \abuse problem" appears if two memory cells store the same (redundant) information. There are at least two solutions to the abuse problem: (1) Sequential network construction (e.g., Fahlman 1991): a memory cell and the corresponding gate units are added to the network whenever the

2For intra-cellular backprop in a quite dierent context see also Doya and Yoshizawa (1989).   
3Following Schmidhuber (1989), we say that a recurrent net algorithm is local in space if the update complexity per time step and weight does not depend on network size. We say that a method is local in time if its storage requirements do not depend on input sequence length. For instance, RTRL is local in time but not in space. BPTT is local in space but not in time.

![](images/f5a64819270880ab2b6fa8089730c4a5d834ea5b825299274965826bae505c23.jpg)  
Figure 2: Example of a net with 8 input units, 4 output units, and 2 memory cel l blocks of size 2. in1 marks the input gate, out1 marks the output gate, and cell1=block1 marks the rst memory cel l of block 1. cell1=block1's architecture is identical to the one in Figure 1, with gate units in1 and out1 (note that by rotating Figure 1 by 90 degrees anti-clockwise, it wil l match with the corresponding parts of Figure 1). The example assumes dense connectivity: each gate unit and each memory cel l see al l non-output units. For simplicity, however, outgoing weights of only one type of unit are shown for each layer. With the ecient, truncated update rule, error ows only through connections to output units, and through xed self-connections within cel l blocks (not shown here | see Figure 1). Error ow is truncated once it \wants" to leave memory cel ls or gate units. Therefore, no connection shown above serves to propagate error back to the unit from which the connection originates (except for connections to output units), although the connections themselves are modiable. That's why the truncated LSTM algorithm is so ecient, despite its ability to bridge very long time lags. See text and appendix A.1 for details. Figure 2 actual ly shows the architecture used for Experiment 6a | only the bias of the non-input units is omitted.

error stops decreasing (see Experiment 2 in Section 5). (2) Output gate bias: each output gate gets a negative initial bias, to push initial memory cell activations towards zero. Memory cells with more negative bias automatically get \allocated" later (see Experiments 1, 3, 4, 5, 6 in Section 5).

Internal state drift and remedies. If memory cell $c _ { j }$ 's inputs are mostly positive or mostly negative, then its internal state $s _ { j }$ will tend to drift away over time. This is potentially dangerous, for the $h ^ { \prime } ( s _ { j } )$ will then adopt very small values, and the gradient will vanish. One way to circumvent this problem is to choose an appropriate function $h$ . But $h ( x ) = x$ , for instance, has the disadvantage of unrestricted memory cell output range. Our simple but eective way of solving drift problems at the beginning of learning is to initially bias the input gate $i n _ { j }$ towards zero. Although there is a tradeo between the magnitudes of $h ^ { \prime } ( s _ { j } )$ on the one hand and of $y ^ { i n _ { j } }$ and $f _ { i n _ { j } } ^ { \prime }$ on the other, the potential negative eect of input gate bias is negligible compared to the one of the drifting eect. With logistic sigmoid activation functions, there appears to be no need for ne-tuning the initial bias, as conrmed by Experiments 4 and 5 in Section 5.4.

# 5 EXPERIMENTS

Introduction. Which tasks are appropriate to demonstrate the quality of a novel long time lag

algorithm? First of all, minimal time lags between relevant input signals and corresponding teacher signals must be long for al l training sequences. In fact, many previous recurrent net algorithms sometimes manage to generalize from very short training sequences to very long test sequences. See, e.g., Pollack (1991). But a real long time lag problem does not have any short time lag exemplars in the training set. For instance, Elman's training procedure, BPTT, oine RTRL, online RTRL, etc., fail miserably on real long time lag problems. See, e.g., Hochreiter (1991) and Mozer (1992). A second important requirement is that the tasks should be complex enough such that they cannot be solved quickly by simple-minded strategies such as random weight guessing.

Guessing can outperform many long time lag algorithms. Recently we discovered (Schmidhuber and Hochreiter 1996, Hochreiter and Schmidhuber 1996, 1997) that many long time lag tasks used in previous work can be solved more quickly by simple random weight guessing than by the proposed algorithms. For instance, guessing solved a variant of Bengio and Frasconi's \parity problem" (1994) problem much faster4 than the seven methods tested by Bengio et al. (1994) and Bengio and Frasconi (1994). Similarly for some of Miller and Giles' problems (1993). Of course, this does not mean that guessing is a good algorithm. It just means that some previously used problems are not extremely appropriate to demonstrate the quality of previously proposed algorithms.

What's common to Experiments 1{6. All our experiments (except for Experiment 1) involve long minimal time lags | there are no short time lag training exemplars facilitating learning. Solutions to most of our tasks are sparse in weight space. They require either many parameters/inputs or high weight precision, such that random weight guessing becomes infeasible.

We always use on-line learning (as opposed to batch learning), and logistic sigmoids as activation functions. For Experiments 1 and 2, initial weights are chosen in the range $[ - 0 . 2 , 0 . 2 ]$ , for the other experiments in $[ - 0 . 1 , 0 . 1 ]$ . Training sequences are generated randomly according to the various task descriptions. In slight deviation from the notation in Appendix A1, each discrete time step of each input sequence involves three processing steps: (1) use current input to set the input units. (2) Compute activations of hidden units (including input gates, output gates, memory cells). (3) Compute output unit activations. Except for Experiments 1, 2a, and 2b, sequence elements are randomly generated on-line, and error signals are generated only at sequence ends. Net activations are reset after each processed input sequence.

For comparisons with recurrent nets taught by gradient descent, we give results only for RTRL, except for comparison 2a, which also includes BPTT. Note, however, that untruncated BPTT (see, e.g., Williams and Peng 1990) computes exactly the same gradient as oine RTRL. With long time lag problems, oine RTRL (or BPTT) and the online version of RTRL (no activation resets, online weight changes) lead to almost identical, negative results (as confirmed by additional simulations in Hochreiter 1991; see also Mozer 1992). This is because oine RTRL, online RTRL, and full BPTT all suer badly from exponential error decay.

Our LSTM architectures are selected quite arbitrarily. If nothing is known about the complexity of a given problem, a more systematic approach would be: start with a very small net consisting of one memory cell. If this does not work, try two cells, etc. Alternatively, use sequential network construction (e.g., Fahlman 1991).

# Outline of experiments.

 Experiment 1 focuses on a standard benchmark test for recurrent nets: the embedded Reber grammar. Since it allows for training sequences with short time lags, it is not a long time lag problem. We include it because (1) it provides a nice example where LSTM's output gates are truly benecial, and (2) it is a popular benchmark for recurrent nets that has been used by many authors | we want to include at least one experiment where conventional BPTT and RTRL do not fail completely (LSTM, however, clearly outperforms them). The embedded Reber grammar's minimal time lags represent a border case in the sense that it is still possible to learn to bridge them with conventional algorithms. Only slightly longer

minimal time lags would make this almost impossible. The more interesting tasks in our paper, however, are those that RTRL, BPTT, etc. cannot solve at all.

 Experiment 2 focuses on noise-free and noisy sequences involving numerous input symbols distracting from the few important ones. The most dicult task (Task 2c) involves hundreds of distractor symbols at random positions, and minimal time lags of 1000 steps. LSTM solves it, while BPTT and RTRL already fail in case of 10-step minimal time lags (see also, e.g., Hochreiter 1991 and Mozer 1992). For this reason RTRL and BPTT are omitted in the remaining, more complex experiments, all of which involve much longer time lags.   
 Experiment 3 addresses long time lag problems with noise and signal on the same input line. Experiments 3a/3b focus on Bengio et al.'s 1994 \2-sequence problem". Because this problem actually can be solved quickly by random weight guessing, we also include a far more dicult 2-sequence problem (3c) which requires to learn real-valued, conditional expectations of noisy targets, given the inputs.   
 Experiments 4 and 5 involve distributed, continuous-valued input representations and require learning to store precise, real values for very long time periods. Relevant input signals can occur at quite dierent positions in input sequences. Again minimal time lags involve hundreds of steps. Similar tasks never have been solved by other recurrent net algorithms.   
 Experiment 6 involves tasks of a dierent complex type that also has not been solved by other recurrent net algorithms. Again, relevant input signals can occur at quite dierent positions in input sequences. The experiment shows that LSTM can extract information conveyed by the temporal order of widely separated inputs.

Subsection 5.7 will provide a detailed summary of experimental conditions in two tables for reference.

# 5.1 EXPERIMENT 1 : EMBEDDED REBER GRAMMAR

Task. Our rst task is to learn the \embedded Reber grammar", e.g. Smith and Zipser (1989), Cleeremans et al. (1989), and Fahlman (1991). Since it allows for training sequences with short time lags (of as few as 9 steps), it is not a long time lag problem. We include it for two reasons: (1) it is a popular recurrent net benchmark used by many authors | we wanted to have at least one experiment where RTRL and BPTT do not fail completely, and (2) it shows nicely how output gates can be beneficial.

![](images/77a4cbaa59f778e84f956669af5c1a81d67de813681878a59ef556154951b8db.jpg)  
Figure 3: Transition diagram for the Reber grammar.   
Figure 4: Transition diagram for the embedded Reber grammar. Each box represents a copy of the Reber grammar (see Figure 3).

Starting at the leftmost node of the directed graph in Figure 4, symbol strings are generated sequentially (beginning with the empty string) by following edges | and appending the associated

symbols to the current string | until the rightmost node is reached. Edges are chosen randomly if there is a choice (probability: 0.5). The net's task is to read strings, one symbol at a time, and to permanently predict the next symbol (error signals occur at every time step). To correctly predict the symbol before last, the net has to remember the second symbol.

Comparison. We compare LSTM to \Elman nets trained by Elman's training procedure" (ELM) (results taken from Cleeremans et al. 1989), Fahlman's \Recurrent Cascade-Correlation" (RCC) (results taken from Fahlman 1991), and RTRL (results taken from Smith and Zipser (1989), where only the few successful trials are listed). It should be mentioned that Smith and Zipser actually make the task easier by increasing the probability of short time lag exemplars. We didn't do this for LSTM.

Training/Testing. We use a local input/output representation (7 input units, 7 output units). Following Fahlman, we use 256 training strings and 256 separate test strings. The training set is generated randomly; training exemplars are picked randomly from the training set. Test sequences are generated randomly, too, but sequences already used in the training set are not used for testing. After string presentation, all activations are reinitialized with zeros. A trial is considered successful if all string symbols of all sequences in both test set and training set are predicted correctly | that is, if the output unit(s) corresponding to the possible next symbol(s) is(are) always the most active ones.

Architectures. Architectures for RTRL, ELM, RCC are reported in the references listed above. For LSTM, we use 3 (4) memory cell blocks. Each block has 2 (1) memory cells. The output layer's only incoming connections originate at memory cells. Each memory cell and each gate unit receives incoming connections from all memory cells and gate units (the hidden layer is fully connected | less connectivity may work as well). The input layer has forward connections to all units in the hidden layer. The gate units are biased. These architecture parameters make it easy to store at least 3 input signals (architectures 3-2 and 4-1 are employed to obtain comparable numbers of weights for both architectures: 264 for 4-1 and 276 for 3-2). Other parameters may be appropriate as well, however. All sigmoid functions are logistic with output range [0; 1], except for $h$ , whose range is $[ - 1 , 1 ]$ , and $g$ , whose range is $[ - 2 , 2 ]$ . All weights are initialized in $[ - 0 . 2 , 0 . 2 ]$ , except for the output gate biases, which are initialized to -1, -2, and -3, respectively (see abuse problem, solution (2) of Section 4). We tried learning rates of 0.1, 0.2 and 0.5.

Results. We use 3 dierent, randomly generated pairs of training and test sets. With each such pair we run 10 trials with dierent initial weights. See Table 1 for results (mean of 30 trials). Unlike the other methods, LSTM always learns to solve the task. Even when we ignore the unsuccessful trials of the other approaches, LSTM learns much faster.

Importance of output gates. The experiment provides a nice example where the output gate is truly benecial. Learning to store the rst T or $\mathrm { P }$ should not perturb activations representing the more easily learnable transitions of the original Reber grammar. This is the job of the output gates. Without output gates, we did not achieve fast learning.

# 5.2 EXPERIMENT 2: NOISE-FREE AND NOISY SEQUENCES

Task 2a: noise-free sequences with long time lags. There are $p + 1$ possible input symbols denoted $a _ { 1 } , . . . , a _ { p - 1 } , a _ { p } = x , a _ { p + 1 } = y$ . $a _ { i }$ is \locally" represented by the $p + 1$ -dimensional vector whose $i$ -th component is 1 (all other components are 0). A net with $p + 1$ input units and $p + 1$ output units sequentially observes input symbol sequences, one at a time, permanently trying to predict the next symbol | error signals occur at every single time step. To emphasize the \long time lag problem", we use a training set consisting of only two very similar sequences: $\left( y , a _ { 1 } , a _ { 2 } , \ldots , a _ { p - 1 } , y \right)$ and $( x , a _ { 1 } , a _ { 2 } , \dotsc , a _ { p - 1 } , x )$ . Each is selected with probability 0.5. To predict the nal element, the net has to learn to store a representation of the rst element for $p$ time steps.

We compare \Real-Time Recurrent Learning" for fully recurrent nets (RTRL), \Back-Propagation Through Time" (BPTT), the sometimes very successful 2-net \Neural Sequence Chunker" (CH, Schmidhuber 1992b), and our new method (LSTM). In all cases, weights are initialized in [-0.2,0.2]. Due to limited computation time, training is stopped after 5 million sequence presen-

Table 1: EXPERIMENT 1: Embedded Reber grammar: percentage of successful trials and number of sequence presentations until success for RTRL (results taken from Smith and Zipser 1989), \Elman net trained by Elman's procedure" (results taken from Cleeremans et al. 1989), \Recurrent Cascade-Correlation" (results taken from Fahlman 1991) and our new approach (LSTM). Weight numbers in the rst 4 rows are estimates | the corresponding papers do not provide al l the technical details. Only LSTM almost always learns to solve the task (only two failures out of 150 trials). Even when we ignore the unsuccessful trials of the other approaches, LSTM learns much faster (the number of required training examples in the bottom row varies between 3,800 and 24,100).   

<table><tr><td>method</td><td>hidden units</td><td># weights</td><td>learning rate</td><td>% of success</td><td>success after</td></tr><tr><td>RTRL</td><td>3</td><td>≈ 170</td><td>0.05</td><td>“some fraction”</td><td>173,000</td></tr><tr><td>RTRL</td><td>12</td><td>≈ 494</td><td>0.1</td><td>“some fraction”</td><td>25,000</td></tr><tr><td>ELM</td><td>15</td><td>≈ 435</td><td></td><td>0</td><td>&gt;200,000</td></tr><tr><td>RCC</td><td>7-9</td><td>≈ 119-198</td><td></td><td>50</td><td>182,000</td></tr><tr><td>LSTM</td><td>4 blocks, size 1</td><td>264</td><td>0.1</td><td>100</td><td>39,740</td></tr><tr><td>LSTM</td><td>3 blocks, size 2</td><td>276</td><td>0.1</td><td>100</td><td>21,730</td></tr><tr><td>LSTM</td><td>3 blocks, size 2</td><td>276</td><td>0.2</td><td>97</td><td>14,060</td></tr><tr><td>LSTM</td><td>4 blocks, size 1</td><td>264</td><td>0.5</td><td>97</td><td>9,500</td></tr><tr><td>LSTM</td><td>3 blocks, size 2</td><td>276</td><td>0.5</td><td>100</td><td>8,440</td></tr></table>

tations. A successful run is one that fullls the following criterion: after training, during 10,000 successive, randomly chosen input sequences, the maximal absolute error of all output units is always below 0:25.

Architectures. RTRL: one self-recurrent hidden unit, $p + 1$ non-recurrent output units. Each layer has connections from all layers below. All units use the logistic activation function sigmoid in [0,1].

BPTT: same architecture as the one trained by RTRL.

CH: both net architectures like RTRL's, but one has an additional output for predicting the hidden unit of the other one (see Schmidhuber 1992b for details).

LSTM: like with RTRL, but the hidden unit is replaced by a memory cell and an input gate (no output gate required). $g$ is the logistic sigmoid, and $h$ is the identity function $h : h ( x ) = x , \forall x$ . Memory cell and input gate are added once the error has stopped decreasing (see abuse problem: solution (1) in Section 4).

Results. Using RTRL and a short 4 time step delay $\mathit { p } = 4$ ), $\frac { 7 } { 9 }$ of all trials were successful. No trial was successful with $p = 1 0$ . With long time lags, only the neural sequence chunker and LSTM achieved successful trials, while BPTT and RTRL failed. With $p = 1 0 0$ , the 2-net sequence chunker solved the task in only $\textstyle { \frac { 1 } { 3 } }$ of all trials. LSTM, however, always learned to solve the task. Comparing successful trials only, LSTM learned much faster. See Table 2 for details. It should be mentioned, however, that a hierarchical chunker can also always quickly solve this task (Schmidhuber 1992c, 1993).

Task 2b: no local regularities. With the task above, the chunker sometimes learns to correctly predict the nal element, but only because of predictable local regularities in the input stream that allow for compressing the sequence. In an additional, more dicult task (involving many more dierent possible sequences), we remove compressibility by replacing the deterministic subsequence $\left( a _ { 1 } , a _ { 2 } , \ldots , a _ { p - 1 } \right)$ by a random subsequence (of length $p - 1 \ r .$ ) over the alphabet $a _ { 1 } , a _ { 2 } , \dotsc , a _ { p - 1 }$ . We obtain 2 classes (two sets of sequences) $\{ ( y , a _ { i _ { 1 } } , a _ { i _ { 2 } } , \dotsc , a _ { i _ { p - 1 } } , y ) \mid 1 \leq$ $i _ { 1 } , i _ { 2 } , \ldots , i _ { p - 1 } \leq p - 1 \}$ and $\{ ( x , a _ { i _ { 1 } } , a _ { i _ { 2 } } , \dotsc , a _ { i _ { p - 1 } } , x ) \mid 1 \leq i _ { 1 } , i _ { 2 } , \dotsc , i _ { p - 1 } \leq p - 1 \}$ . Again, every next sequence element has to be predicted. The only totally predictable targets, however, are $x$ and $y$ , which occur at sequence ends. Training exemplars are chosen randomly from the 2 classes. Architectures and parameters are the same as in Experiment 2a. A successful run is one that fullls the following criterion: after training, during 10,000 successive, randomly chosen input

Table 2: Task 2a: Percentage of successful trials and number of training sequences until success, for \Real-Time Recurrent Learning" (RTRL), \Back-Propagation Through Time" (BPTT), neural sequence chunking (CH), and the new method (LSTM). Table entries refer to means of 18 trials. With 100 time step delays, only CH and LSTM achieve successful trials. Even when we ignore the unsuccessful trials of the other approaches, LSTM learns much faster.   

<table><tr><td>Method</td><td>Delay p</td><td>Learning rate</td><td># weights</td><td>% Successful trials</td><td>Success after</td></tr><tr><td>RTRL</td><td>4</td><td>1.0</td><td>36</td><td>78</td><td>1,043,000</td></tr><tr><td>RTRL</td><td>4</td><td>4.0</td><td>36</td><td>56</td><td>892,000</td></tr><tr><td>RTRL</td><td>4</td><td>10.0</td><td>36</td><td>22</td><td>254,000</td></tr><tr><td>RTRL</td><td>10</td><td>1.0-10.0</td><td>144</td><td>0</td><td>&gt;5,000,000</td></tr><tr><td>RTRL</td><td>100</td><td>1.0-10.0</td><td>10404</td><td>0</td><td>&gt;5,000,000</td></tr><tr><td>BPTT</td><td>100</td><td>1.0-10.0</td><td>10404</td><td>0</td><td>&gt;5,000,000</td></tr><tr><td>CH</td><td>100</td><td>1.0</td><td>10506</td><td>33</td><td>32,400</td></tr><tr><td>LSTM</td><td>100</td><td>1.0</td><td>10504</td><td>100</td><td>5,040</td></tr></table>

sequences, the maximal absolute error of all output units is below 0:25 at sequence end.

Results. As expected, the chunker failed to solve this task (so did BPTT and RTRL, of course). LSTM, however, was always successful. On average (mean of 18 trials), success for $p = 1 0 0$ was achieved after 5,680 sequence presentations. This demonstrates that LSTM does not require sequence regularities to work well.

Task 2c: very long time lags | no local regularities. This is the most dicult task in this subsection. To our knowledge no other recurrent net algorithm can solve it. Now there are $p { + 4 }$ possible input symbols denoted $a _ { 1 } , . . . , a _ { p - 1 } , a _ { p } , a _ { p + 1 } = e , a _ { p + 2 } = b , a _ { p + 3 } = x , a _ { p + 4 } = y . a _ { 1 } , . . . ,$ $a _ { p + 1 } = e$ $a _ { 1 } , . . . , a _ { p }$ are also called \distractor symbols". Again, $a _ { i }$ is locally represented by the $p { + 4 }$ -dimensional vector whose ith component is 1 (all other components are 0). A net with $p + 4$ input units and 2 output units sequentially observes input symbol sequences, one at a time. Training sequences are randomly chosen from the union of two very similar subsets of sequences: $\{ ( b , y , a _ { i _ { 1 } } , a _ { i _ { 2 } } , \ldots , a _ { i _ { q + k } } , e , y ) \mid 1 \leq$ $i _ { 1 } , i _ { 2 } , \ldots , i _ { q + k } \le q \}$ and $\{ ( b , x , a _ { i _ { 1 } } , a _ { i _ { 2 } } , \ldots , a _ { i _ { q + k } } , e , x ) | 1 \leq i _ { 1 } , i _ { 2 } , \ldots , i _ { q + k } \leq q \}$ . To produce a training sequence, we (1) randomly generate a sequence prex of length $q + 2$ , (2) randomly generate a sequence sux of additional elements $( \neq b , e , x , y )$ with probability $\frac { 9 } { 1 0 }$ or, alternatively, an $e$ with probability $\textstyle { \frac { 1 } { 1 0 } }$ . In the latter case, we (3) conclude the sequence with $x$ or $y$ , depending on the second element. For a given $k$ , this leads to a uniform distribution on the possible sequences with length $q + k + 4$ . The minimal sequence length is $q + 4$ ; the expected length is

$$
4 + \sum_ {k = 0} ^ {\infty} \frac {1}{1 0} (\frac {9}{1 0}) ^ {k} (q + k) = q + 1 4.
$$

The expected number of occurrences of element $a _ { i } , 1 \leq i \leq p$ , in a sequence is $\begin{array} { r } { \frac { q + 1 0 } { p _ { - } } \approx \frac { q } { p _ { - } } } \end{array}$ . The goal is to predict the last symbol, which always occurs after the \trigger symbol" $e$ p p. Error signals are generated only at sequence ends. To predict the nal element, the net has to learn to store a representation of the second element for at least $q + 1$ time steps (until it sees the trigger symbol e). Success is dened as \prediction error (for nal sequence element) of both output units always below 0:2, for 10,000 successive, randomly chosen input sequences".

Architecture/Learning. The net has $p + 4$ input units and 2 output units. Weights are initialized in [-0.2,0.2]. To avoid too much learning time variance due to dierent weight initializations, the hidden layer gets two memory cells (two cell blocks of size 1 | although one would be sucient). There are no other hidden units. The output layer receives connections only from memory cells. Memory cells and gate units receive connections from input units, memory cells and gate units (i.e., the hidden layer is fully connected). No bias weights are used. h and $g$ are logistic sigmoids with output ranges $[ - 1 , 1 ]$ and $[ - 2 , 2 ]$ , respectively. The learning rate is 0.01.

Table 3: Task 2c: LSTM with very long minimal time lags $q + 1$ and a lot of noise. p is the number of available distractor symbols $\mathit { p } + 4$ is the number of input units). $\frac { q } { p }$ is the expected number of occurrences of a given distractor symbol in a sequence. The rightmost column lists the number of training sequences required by LSTM (BPTT, RTRL and the other competitors have no chance of solving this task). If we let the number of distractor symbols (and weights) increase in proportion to the time lag, learning time increases very slow ly. The lower block il lustrates the expected slow-down due to increased frequency of distractor symbols.   

<table><tr><td>q (time lag -1)</td><td>p (# random inputs)</td><td>q p</td><td># weights</td><td>Success after</td></tr><tr><td>50</td><td>50</td><td>1</td><td>364</td><td>30,000</td></tr><tr><td>100</td><td>100</td><td>1</td><td>664</td><td>31,000</td></tr><tr><td>200</td><td>200</td><td>1</td><td>1264</td><td>33,000</td></tr><tr><td>500</td><td>500</td><td>1</td><td>3064</td><td>38,000</td></tr><tr><td>1,000</td><td>1,000</td><td>1</td><td>6064</td><td>49,000</td></tr><tr><td>1,000</td><td>500</td><td>2</td><td>3064</td><td>49,000</td></tr><tr><td>1,000</td><td>200</td><td>5</td><td>1264</td><td>75,000</td></tr><tr><td>1,000</td><td>100</td><td>10</td><td>664</td><td>135,000</td></tr><tr><td>1,000</td><td>50</td><td>20</td><td>364</td><td>203,000</td></tr></table>

Note that the minimal time lag is $q + 1$ | the net never sees short training sequences facilitating the classication of long test sequences.

Results. 20 trials were made for all tested pairs $( p , q )$ . Table 3 lists the mean of the number of training sequences required by LSTM to achieve success (BPTT and RTRL have no chance of solving non-trivial tasks with minimal time lags of 1000 steps).

Scaling. Table 3 shows that if we let the number of input symbols (and weights) increase in proportion to the time lag, learning time increases very slowly. This is a another remarkable property of LSTM not shared by any other method we are aware of. Indeed, RTRL and BPTT are far from scaling reasonably | instead, they appear to scale exponentially, and appear quite useless when the time lags exceed as few as 10 steps.

Distractor inuence. In Table 3, the column headed by $\frac { q } { p }$ gives the expected frequency of distractor symbols. Increasing this frequency decreases learning speed, an eect due to weight oscillations caused by frequently observed input symbols.

# 5.3 EXPERIMENT 3: NOISE AND SIGNAL ON SAME CHANNEL

This experiment serves to illustrate that LSTM does not encounter fundamental problems if noise and signal are mixed on the same input line. We initially focus on Bengio et al.'s simple 1994 \2-sequence problem"; in Experiment 3c we will then pose a more challenging 2-sequence problem.

Task 3a (\2-sequence problem"). The task is to observe and then classify input sequences. There are two classes, each occurring with probability 0.5. There is only one input line. Only the rst N real-valued sequence elements convey relevant information about the class. Sequence elements at positions $t > N$ are generated by a Gaussian with mean zero and variance 0.2. Case $N = 1$ : the rst sequence element is 1.0 for class 1, and -1.0 for class 2. Case $N = 3$ : the rst three elements are 1.0 for class 1 and -1.0 for class 2. The target at the sequence end is 1.0 for class 1 and 0.0 for class 2. Correct classication is dened as \absolute output error at sequence end below $0 . 2 ^ { \mathfrak { n } }$ . Given a constant T, the sequence length is randomly selected between T and $\mathrm { ~ T ~ } +$ $\mathrm { T / 1 0 }$ (a dierence to Bengio et al.'s problem is that they also permit shorter sequences of length $\mathrm { T } / 2$ ).

Guessing. Bengio et al. (1994) and Bengio and Frasconi (1994) tested 7 dierent methods on the 2-sequence problem. We discovered, however, that random weight guessing easily outper-

Table 4: Task 3a: Bengio et al.'s 2-sequence problem. T is minimal sequence length. N is the number of information-conveying elements at sequence begin. The column headed by ST1 (ST2) gives the number of sequence presentations required to achieve stopping criterion ST1 (ST2). The rightmost column lists the fraction of misclassied post-training sequences (with absolute error > 0.2) from a test set consisting of 2560 sequences (tested after ST2 was achieved). Al l values are means of 10 trials. We discovered, however, that this problem is so simple that random weight guessing solves it faster than LSTM and any other method for which there are published results.   

<table><tr><td>T</td><td>N</td><td>stop: ST1</td><td>stop: ST2</td><td># weights</td><td>ST2: fraction misclassified</td></tr><tr><td>100</td><td>3</td><td>27,380</td><td>39,850</td><td>102</td><td>0.000195</td></tr><tr><td>100</td><td>1</td><td>58,370</td><td>64,330</td><td>102</td><td>0.000117</td></tr><tr><td>1000</td><td>3</td><td>446,850</td><td>452,460</td><td>102</td><td>0.000078</td></tr></table>

forms them all, because the problem is so simple5. See Schmidhuber and Hochreiter (1996) and Hochreiter and Schmidhuber (1996, 1997) for additional results in this vein.

LSTM architecture. We use a 3-layer net with 1 input unit, 1 output unit, and 3 cell blocks of size 1. The output layer receives connections only from memory cells. Memory cells and gate units receive inputs from input units, memory cells and gate units, and have bias weights. Gate units and output unit are logistic sigmoid in [0; 1], $h$ in $[ - 1 , 1 ]$ , and $g$ in $[ - 2 , 2 ]$ .

Training/Testing. All weights (except the bias weights to gate units) are randomly initialized in the range $[ - 0 . 1 , 0 . 1 ]$ ]. The rst input gate bias is initialized with $- 1 . 0$ , the second with $- 3 . 0$ , and the third with $- 5 . 0$ . The rst output gate bias is initialized with $- 2 . 0$ , the second with  4:0 and the third with  6:0. The precise initialization values hardly matter though, as conrmed by additional experiments. The learning rate is 1.0. All activations are reset to zero at the beginning of a new sequence.

We stop training (and judge the task as being solved) according to the following criteria: ST1: none of 256 sequences from a randomly chosen test set is misclassied. ST2: ST1 is satised, and mean absolute test set error is below 0.01. In case of ST2, an additional test set consisting of 2560 randomly chosen sequences is used to determine the fraction of misclassied sequences.

Results. See Table 4. The results are means of 10 trials with dierent weight initializations in the range $[ - 0 . 1 , 0 . 1 ]$ . LSTM is able to solve this problem, though by far not as fast as random weight guessing (see paragraph \Guessing" above). Clearly, this trivial problem does not provide a very good testbed to compare performance of various non-trivial algorithms. Still, it demonstrates that LSTM does not encounter fundamental problems when faced with signal and noise on the same channel.

Task 3b. Architecture, parameters, etc. like in Task 3a, but now with Gaussian noise (mean 0 and variance 0.2) added to the information-conveying elements ( $t < = N$ ). We stop training (and judge the task as being solved) according to the following, slightly redefined criteria: ST1: less than 6 out of 256 sequences from a randomly chosen test set are misclassied. ST2: ST1 is satisfied, and mean absolute test set error is below 0.04. In case of ST2, an additional test set consisting of 2560 randomly chosen sequences is used to determine the fraction of misclassied sequences.

Results. See Table 5. The results represent means of 10 trials with dierent weight initializations. LSTM easily solves the problem.

Task 3c. Architecture, parameters, etc. like in Task 3a, but with a few essential changes that make the task non-trivial: the targets are 0.2 and 0.8 for class 1 and class 2, respectively, and there is Gaussian noise on the targets (mean 0 and variance 0.1; st.dev. 0.32). To minimize mean squared error, the system has to learn the conditional expectations of the targets given the inputs. Misclassication is dened as \absolute dierence between output and noise-free target (0.2 for

Table 5: Task 3b: modied 2-sequence problem. Same as in Table 4, but now the informationconveying elements are also perturbed by noise.   

<table><tr><td>T</td><td>N</td><td>stop: ST1</td><td>stop: ST2</td><td># weights</td><td>ST2: fraction misclassified</td></tr><tr><td>100</td><td>3</td><td>41,740</td><td>43,250</td><td>102</td><td>0.00828</td></tr><tr><td>100</td><td>1</td><td>74,950</td><td>78,430</td><td>102</td><td>0.01500</td></tr><tr><td>1000</td><td>1</td><td>481,060</td><td>485,080</td><td>102</td><td>0.01207</td></tr></table>

Table 6: Task 3c: modied, more chal lenging 2-sequence problem. Same as in Table 4, but with noisy real-valued targets. The system has to learn the conditional expectations of the targets given the inputs. The rightmost column provides the average dierence between network output and expected target. Unlike 3a and 3b, this task cannot be solved quickly by random weight guessing.   

<table><tr><td>T</td><td>N</td><td>stop</td><td># weights</td><td>fraction misclassified</td><td>av. difference to mean</td></tr><tr><td>100</td><td>3</td><td>269,650</td><td>102</td><td>0.00558</td><td>0.014</td></tr><tr><td>100</td><td>1</td><td>565,640</td><td>102</td><td>0.00441</td><td>0.012</td></tr></table>

class 1 and 0.8 for class 2) > 0.1. " The network output is considered acceptable if the mean absolute dierence between noise-free target and output is below 0.015. Since this requires high weight precision, Task 3c (unlike 3a and 3b) cannot be solved quickly by random guessing.

Training/Testing. The learning rate is 0:1. We stop training according to the following criterion: none of 256 sequences from a randomly chosen test set is misclassied, and mean absolute difference between noise free target and output is below 0.015. An additional test set consisting of 2560 randomly chosen sequences is used to determine the fraction of misclassied sequences.

Results. See Table 6. The results represent means of 10 trials with dierent weight initializations. Despite the noisy targets, LSTM still can solve the problem by learning the expected target values.

# 5.4 EXPERIMENT 4: ADDING PROBLEM

The dicult task in this section is of a type that has never been solved by other recurrent net algorithms. It shows that LSTM can solve long time lag problems involving distributed, continuousvalued representations.

Task. Each element of each input sequence is a pair of components. The rst component is a real value randomly chosen from the interval $[ - 1 , 1 ]$ ; the second is either 1.0, 0.0, or -1.0, and is used as a marker: at the end of each sequence, the task is to output the sum of the rst components of those pairs that are marked by second components equal to 1.0. Sequences have random lengths between the minimal sequence length $T$ and $\begin{array} { r } { T + \frac { T } { 1 0 } } \end{array}$ . In a given sequence exactly two pairs are marked as follows: we rst randomly select and mark one of the rst ten pairs (whose rst component we call $X _ { 1 }$ ). Then we randomly select and mark one of the rst $\begin{array} { r } { \frac { T } { 2 } - 1 } \end{array}$ still unmarked pairs (whose rst component we call $X _ { 2 }$ ). The second components of all remaining pairs are zero except for the rst and nal pair, whose second components are -1. (In the rare case where the rst pair of the sequence gets marked, we set $X _ { 1 }$ to zero.) An error signal is generated only at the sequence end: the target is $0 . 5 + \frac { X _ { 1 } + X _ { 2 } } { 4 . 0 }$ 1 (the sum $X _ { 1 } + X _ { 2 }$ scaled to the interval [0; 1]). A sequence is processed correctly if the absolute error at the sequence end is below 0.04.

Architecture. We use a 3-layer net with 2 input units, 1 output unit, and 2 cell blocks of size 2. The output layer receives connections only from memory cells. Memory cells and gate units receive inputs from memory cells and gate units (i.e., the hidden layer is fully connected | less connectivity may work as well). The input layer has forward connections to all units in the hidden

Table 7: EXPERIMENT 4: Results for the Adding Problem. T is the minimal sequence length, $T / 2$ the minimal time lag. \# wrong predictions" is the number of incorrectly processed sequences (error > 0.04) from a test set containing 2560 sequences. The rightmost column gives the number of training sequences required to achieve the stopping criterion. Al l values are means of 10 trials. For $T = 1 0 0 0$ the number of required training examples varies between 370,000 and 2,020,000, exceeding 700,000 in only 3 cases.   

<table><tr><td>T</td><td>minimal lag</td><td># weights</td><td># wrong predictions</td><td>Success after</td></tr><tr><td>100</td><td>50</td><td>93</td><td>1 out of 2560</td><td>74,000</td></tr><tr><td>500</td><td>250</td><td>93</td><td>0 out of 2560</td><td>209,000</td></tr><tr><td>1000</td><td>500</td><td>93</td><td>1 out of 2560</td><td>853,000</td></tr></table>

layer. All non-input units have bias weights. These architecture parameters make it easy to store at least 2 input signals (a cell block size of 1 works well, too). All activation functions are logistic with output range [0; 1], except for $h$ , whose range is $[ - 1 , 1 ]$ , and $g$ , whose range is $[ - 2 , 2 ]$ .

State drift versus initial bias. Note that the task requires storing the precise values of real numbers for long durations | the system must learn to protect memory cell contents against even minor internal state drift (see Section 4). To study the signicance of the drift problem, we make the task even more dicult by biasing all non-input units, thus articially inducing internal state drift. All weights (including the bias weights) are randomly initialized in the range [ 0:1; 0:1]. Following Section 4's remedy for state drifts, the rst input gate bias is initialized with  3:0, the second with $- 6 . 0$ (though the precise values hardly matter, as conrmed by additional experiments).

Training/Testing. The learning rate is 0.5. Training is stopped once the average training error is below 0.01, and the 2000 most recent sequences were processed correctly.

Results. With a test set consisting of 2560 randomly chosen sequences, the average test set error was always below 0.01, and there were never more than 3 incorrectly processed sequences. Table 7 shows details.

The experiment demonstrates: (1) LSTM is able to work well with distributed representations. (2) LSTM is able to learn to perform calculations involving continuous values. (3) Since the system manages to store continuous values without deterioration for minimal delays of $\begin{array} { l } { { \frac { T } { 2 } } } \end{array}$ time steps, there is no signicant, harmful internal state drift.

# 5.5 EXPERIMENT 5: MULTIPLICATION PROBLEM

One may argue that LSTM is a bit biased towards tasks such as the Adding Problem from the previous subsection. Solutions to the Adding Problem may exploit the CEC's built-in integration capabilities. Although this CEC property may be viewed as a feature rather than a disadvantage (integration seems to be a natural subtask of many tasks occurring in the real world), the question arises whether LSTM can also solve tasks with inherently non-integrative solutions. To test this, we change the problem by requiring the nal target to equal the product (instead of the sum) of earlier marked inputs.

Task. Like the task in Section 5.4, except that the rst component of each pair is a real value randomly chosen from the interval [0; 1]. In the rare case where the rst pair of the input sequence gets marked, we set $X _ { 1 }$ to 1.0. The target at sequence end is the product $X _ { 1 } \times X _ { 2 }$ .

Architecture. Like in Section 5.4. All weights (including the bias weights) are randomly initialized in the range $[ - 0 . 1 , 0 . 1 ]$ .

Training/Testing. The learning rate is 0.1. We test performance twice: as soon as less than $n _ { s e q }$ of the 2000 most recent training sequences lead to absolute errors exceeding 0.04, where $n _ { s e q } = 1 4 0$ , and $n _ { s e q } = 1 3$ . Why these values? $n _ { s e q } = 1 4 0$ is sucient to learn storage of the relevant inputs. It is not enough though to ne-tune the precise nal outputs. $n _ { s e q } = 1 3$ , however,

Table 8: EXPERIMENT 5: Results for the Multiplication Problem. T is the minimal sequence length, T =2 the minimal time lag. We test on a test set containing 2560 sequences as soon as less than nseq of the 2000 most recent training sequences lead to error > 0.04. \# wrong predictions" is the number of test sequences with error $> ~ 0 . 0 4$ . MSE is the mean squared error on the test set. The rightmost column lists numbers of training sequences required to achieve the stopping criterion. Al l values are means of 10 trials.   

<table><tr><td>T</td><td>minimal lag</td><td># weights</td><td>n_seq</td><td># wrong predictions</td><td>MSE</td><td>Success after</td></tr><tr><td>100</td><td>50</td><td>93</td><td>140</td><td>139 out of 2560</td><td>0.0223</td><td>482,000</td></tr><tr><td>100</td><td>50</td><td>93</td><td>13</td><td>14 out of 2560</td><td>0.0139</td><td>1,273,000</td></tr></table>

leads to quite satisfactory results.

Results. For $n _ { s e q } ~ = ~ 1 4 0$ $_ { n _ { s e q } } = 1 3$ ) with a test set consisting of 2560 randomly chosen sequences, the average test set error was always below 0.026 (0.013), and there were never more than 170 (15) incorrectly processed sequences. Table 8 shows details. (A net with additional standard hidden units or with a hidden layer above the memory cells may learn the ne-tuning part more quickly.)

The experiment demonstrates: LSTM can solve tasks involving both continuous-valued representations and non-integrative information processing.

# 5.6 EXPERIMENT 6: TEMPORAL ORDER

In this subsection, LSTM solves other dicult (but articial) tasks that have never been solved by previous recurrent net algorithms. The experiment shows that LSTM is able to extract information conveyed by the temporal order of widely separated inputs.

Task 6a: two relevant, widely separated symbols. The goal is to classify sequences. Elements and targets are represented locally (input vectors with only one non-zero bit). The sequence starts with an $E$ , ends with a $B$ (the \trigger symbol") and otherwise consists of randomly chosen symbols from the set $\{ a , b , c , d \}$ except for two elements at positions $t _ { 1 }$ and $t _ { 2 }$ that are either $X$ or $Y$ . The sequence length is randomly chosen between 100 and 110, $t _ { 1 }$ is randomly chosen between 10 and 20, and $t _ { 2 }$ is randomly chosen between 50 and 60. There are 4 sequence classes $Q , R , S , U$ which depend on the temporal order of $X$ and $Y$ . The rules are: $X , X  Q$ ; $X , Y $ $R$ ; $Y , X  S$ ; $Y , Y  U$ .

Task 6b: three relevant, widely separated symbols. Again, the goal is to classify sequences. Elements/targets are represented locally. The sequence starts with an $E$ , ends with a $B$ (the \trigger symbol"), and otherwise consists of randomly chosen symbols from the set $\{ a , b , c , d \}$ except for three elements at positions $t _ { 1 } , t _ { 2 }$ and $t _ { 3 }$ that are either $X$ or $Y$ . The sequence length is randomly chosen between 100 and 110, $t _ { 1 }$ is randomly chosen between 10 and 20, $t _ { 2 }$ is randomly chosen between 33 and 43, and $t _ { 3 }$ is randomly chosen between 66 and 76. There are 8 sequence classes $Q , R , S , U , V , A , B , C$ which depend on the temporal order of the $X \mathrm { s }$ s and $Y \mathrm { s }$ . The rules are: $X , X , X  Q$ ; $X , X , Y  R$ ; $X , Y , X  S$ ; $X , Y , Y  U$ ; $Y , X , X  V$ ; $Y , X , Y $ A; Y ; Y ; X ! B; $Y , Y , Y  C$ .

There are as many output units as there are classes. Each class is locally represented by a binary target vector with one non-zero component. With both tasks, error signals occur only at the end of a sequence. The sequence is classied correctly if the nal absolute error of all output units is below 0.3.

Architecture. We use a 3-layer net with 8 input units, 2 (3) cell blocks of size 2 and 4 (8) output units for Task 6a (6b). Again all non-input units have bias weights, and the output layer receives connections from memory cells only. Memory cells and gate units receive inputs from input units, memory cells and gate units (i.e., the hidden layer is fully connected | less connectivity may work as well). The architecture parameters for Task 6a (6b) make it easy to

store at least 2 (3) input signals. All activation functions are logistic with output range [0; 1], except for $h$ , whose range is $[ - 1 , 1 ]$ , and $g$ , whose range is $[ - 2 , 2 ]$ .

Training/Testing. The learning rate is 0.5 (0.1) for Experiment 6a (6b). Training is stopped once the average training error falls below 0.1 and the 2000 most recent sequences were classied correctly. All weights are initialized in the range $[ - 0 . 1 , 0 . 1 ]$ . The rst input gate bias is initialized with $- 2 . 0$ , the second with  4:0, and (for Experiment 6b) the third with $- 6 . 0$ (again, we conrmed by additional experiments that the precise values hardly matter).

Results. With a test set consisting of 2560 randomly chosen sequences, the average test set error was always below 0.1, and there were never more than 3 incorrectly classied sequences. Table 9 shows details.

The experiment shows that LSTM is able to extract information conveyed by the temporal order of widely separated inputs. In Task 6a, for instance, the delays between rst and second relevant input and between second relevant input and sequence end are at least 30 time steps.

Table 9: EXPERIMENT 6: Results for the Temporal Order Problem. \# wrong predictions" is the number of incorrectly classied sequences (error > 0.3 for at least one output unit) from a test set containing 2560 sequences. The rightmost column gives the number of training sequences required to achieve the stopping criterion. The results for Task 6a are means of 20 trials; those for Task 6b of 10 trials.   

<table><tr><td>task</td><td># weights</td><td># wrong predictions</td><td>Success after</td></tr><tr><td>Task 6a</td><td>156</td><td>1 out of 2560</td><td>31,390</td></tr><tr><td>Task 6b</td><td>308</td><td>2 out of 2560</td><td>571,100</td></tr></table>

Typical solutions. In Experiment 6a, how does LSTM distinguish between temporal orders $( X , Y )$ and $( Y , X )$ ? One of many possible solutions is to store the first $X$ or $Y$ in cell block 1, and the second $X / Y$ in cell block 2. Before the rst $X / Y$ occurs, block 1 can see that it is still empty by means of its recurrent connections. After the rst $X / Y$ , block 1 can close its input gate. Once block 1 is lled and closed, this fact will become visible to block 2 (recall that all gate units and all memory cells receive connections from all non-output units).

Typical solutions, however, require only one memory cell block. The block stores the rst $X$ or $Y$ ; once the second $X / Y$ occurs, it changes its state depending on the rst stored symbol. Solution type 1 exploits the connection between memory cell output and input gate unit | the following events cause dierent input gate activations: \ $X$ occurs in conjunction with a lled block"; \ $X$ occurs in conjunction with an empty block". Solution type 2 is based on a strong positive connection between memory cell output and memory cell input. The previous occurrence of $X$ (Y ) is represented by a positive (negative) internal state. Once the input gate opens for the second time, so does the output gate, and the memory cell output is fed back to its own input. This causes $( X , Y )$ to be represented by a positive internal state, because $X$ contributes to the new internal state twice (via current internal state and cell output feedback). Similarly, $( Y , X )$ gets represented by a negative internal state.

# 5.7 SUMMARY OF EXPERIMENTAL CONDITIONS

The two tables in this subsection provide an overview of the most important LSTM parameters and architectural details for Experiments 1{6. The conditions of the simple experiments 2a and 2b dier slightly from those of the other, more systematic experiments, due to historical reasons.

<table><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td><td>15</td></tr><tr><td>Task</td><td>p</td><td>lag</td><td>b</td><td>s</td><td>in</td><td>out</td><td>w</td><td>c</td><td>ogb</td><td>igb</td><td>bias</td><td>h</td><td>g</td><td>α</td></tr><tr><td>1-1</td><td>9</td><td>9</td><td>4</td><td>1</td><td>7</td><td>7</td><td>264</td><td>F</td><td>-1,-2,-3,-4</td><td>r</td><td>ga</td><td>h1</td><td>g2</td><td>0.1</td></tr><tr><td>1-2</td><td>9</td><td>9</td><td>3</td><td>2</td><td>7</td><td>7</td><td>276</td><td>F</td><td>-1,-2,-3</td><td>r</td><td>ga</td><td>h1</td><td>g2</td><td>0.1</td></tr><tr><td colspan="15">to be continued on next page</td></tr></table>

<table><tr><td colspan="15">continued from previous page</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td><td>15</td></tr><tr><td>Task</td><td>p</td><td>lag</td><td>b</td><td>s</td><td>in</td><td>out</td><td>w</td><td>c</td><td>ogb</td><td>igb</td><td>bias</td><td>h</td><td>g</td><td>α</td></tr><tr><td>1-3</td><td>9</td><td>9</td><td>3</td><td>2</td><td>7</td><td>7</td><td>276</td><td>F</td><td>-1,-2,-3</td><td>r</td><td>ga</td><td>h1</td><td>g2</td><td>0.2</td></tr><tr><td>1-4</td><td>9</td><td>9</td><td>4</td><td>1</td><td>7</td><td>7</td><td>264</td><td>F</td><td>-1,-2,-3,-4</td><td>r</td><td>ga</td><td>h1</td><td>g2</td><td>0.5</td></tr><tr><td>1-5</td><td>9</td><td>9</td><td>3</td><td>2</td><td>7</td><td>7</td><td>276</td><td>F</td><td>-1,-2,-3</td><td>r</td><td>ga</td><td>h1</td><td>g2</td><td>0.5</td></tr><tr><td>2a</td><td>100</td><td>100</td><td>1</td><td>1</td><td>101</td><td>101</td><td>10504</td><td>B</td><td>no og</td><td>none</td><td>none</td><td>id</td><td>g1</td><td>1.0</td></tr><tr><td>2b</td><td>100</td><td>100</td><td>1</td><td>1</td><td>101</td><td>101</td><td>10504</td><td>B</td><td>no og</td><td>none</td><td>none</td><td>id</td><td>g1</td><td>1.0</td></tr><tr><td>2c-1</td><td>50</td><td>50</td><td>2</td><td>1</td><td>54</td><td>2</td><td>364</td><td>F</td><td>none</td><td>none</td><td>none</td><td>h1</td><td>g2</td><td>0.01</td></tr><tr><td>2c-2</td><td>100</td><td>100</td><td>2</td><td>1</td><td>104</td><td>2</td><td>664</td><td>F</td><td>none</td><td>none</td><td>none</td><td>h1</td><td>g2</td><td>0.01</td></tr><tr><td>2c-3</td><td>200</td><td>200</td><td>2</td><td>1</td><td>204</td><td>2</td><td>1264</td><td>F</td><td>none</td><td>none</td><td>none</td><td>h1</td><td>g2</td><td>0.01</td></tr><tr><td>2c-4</td><td>500</td><td>500</td><td>2</td><td>1</td><td>504</td><td>2</td><td>3064</td><td>F</td><td>none</td><td>none</td><td>none</td><td>h1</td><td>g2</td><td>0.01</td></tr><tr><td>2c-5</td><td>1000</td><td>1000</td><td>2</td><td>1</td><td>1004</td><td>2</td><td>6064</td><td>F</td><td>none</td><td>none</td><td>none</td><td>h1</td><td>g2</td><td>0.01</td></tr><tr><td>2c-6</td><td>1000</td><td>1000</td><td>2</td><td>1</td><td>504</td><td>2</td><td>3064</td><td>F</td><td>none</td><td>none</td><td>none</td><td>h1</td><td>g2</td><td>0.01</td></tr><tr><td>2c-7</td><td>1000</td><td>1000</td><td>2</td><td>1</td><td>204</td><td>2</td><td>1264</td><td>F</td><td>none</td><td>none</td><td>none</td><td>h1</td><td>g2</td><td>0.01</td></tr><tr><td>2c-8</td><td>1000</td><td>1000</td><td>2</td><td>1</td><td>104</td><td>2</td><td>664</td><td>F</td><td>none</td><td>none</td><td>none</td><td>h1</td><td>g2</td><td>0.01</td></tr><tr><td>2c-9</td><td>1000</td><td>1000</td><td>2</td><td>1</td><td>54</td><td>2</td><td>364</td><td>F</td><td>none</td><td>none</td><td>none</td><td>h1</td><td>g2</td><td>0.01</td></tr><tr><td>3a</td><td>100</td><td>100</td><td>3</td><td>1</td><td>1</td><td>1</td><td>102</td><td>F</td><td>-2,-4,-6</td><td>-1,-3,-5</td><td>b1</td><td>h1</td><td>g2</td><td>1.0</td></tr><tr><td>3b</td><td>100</td><td>100</td><td>3</td><td>1</td><td>1</td><td>1</td><td>102</td><td>F</td><td>-2,-4,-6</td><td>-1,-3,-5</td><td>b1</td><td>h1</td><td>g2</td><td>1.0</td></tr><tr><td>3c</td><td>100</td><td>100</td><td>3</td><td>1</td><td>1</td><td>1</td><td>102</td><td>F</td><td>-2,-4,-6</td><td>-1,-3,-5</td><td>b1</td><td>h1</td><td>g2</td><td>0.1</td></tr><tr><td>4-1</td><td>100</td><td>50</td><td>2</td><td>2</td><td>2</td><td>1</td><td>93</td><td>F</td><td>r</td><td>-3,-6</td><td>all</td><td>h1</td><td>g2</td><td>0.5</td></tr><tr><td>4-2</td><td>500</td><td>250</td><td>2</td><td>2</td><td>2</td><td>1</td><td>93</td><td>F</td><td>r</td><td>-3,-6</td><td>all</td><td>h1</td><td>g2</td><td>0.5</td></tr><tr><td>4-3</td><td>1000</td><td>500</td><td>2</td><td>2</td><td>2</td><td>1</td><td>93</td><td>F</td><td>r</td><td>-3,-6</td><td>all</td><td>h1</td><td>g2</td><td>0.5</td></tr><tr><td>5</td><td>100</td><td>50</td><td>2</td><td>2</td><td>2</td><td>1</td><td>93</td><td>F</td><td>r</td><td>r</td><td>all</td><td>h1</td><td>g2</td><td>0.1</td></tr><tr><td>6a</td><td>100</td><td>40</td><td>2</td><td>2</td><td>8</td><td>4</td><td>156</td><td>F</td><td>r</td><td>-2,-4</td><td>all</td><td>h1</td><td>g2</td><td>0.5</td></tr><tr><td>6b</td><td>100</td><td>24</td><td>3</td><td>2</td><td>8</td><td>8</td><td>308</td><td>F</td><td>r</td><td>-2,-4,-6</td><td>all</td><td>h1</td><td>g2</td><td>0.1</td></tr></table>

Table 10: Summary of experimental conditions for LSTM, Part I. 1st column: task number. 2nd column: minimal sequence length $p$ . 3rd column: minimal number of steps between most recent relevant input information and teacher signal. 4th column: number of cel l blocks b. 5th column: block size s. 6th column: number of input units in. 7th column: number of output units out. 8th column: number of weights w. 9th column: c describes connectivity: \F" means \output layer receives connections from memory cel ls; memory cel ls and gate units receive connections from input units, memory cel ls and gate units"; \B" means \each layer receives connections from al l layers below". 10th column: initial output gate bias ogb, where \r" stands for \randomly chosen from the interval [ 0:1; 0:1]" and \no og" means \no output gate used". 11th column: initial input gate bias igb (see 10th column). 12th column: which units have bias weights? \b1" stands for \al l hidden units", \ga" for \only gate units", and \al l" for \al l non-input units". 13th column: the function h, where \id" is identity function, \h1" is logistic sigmoid in [ 2; 2]. 14th column: the logistic function g, where \g1" is sigmoid in [0; 1], \g2" in [ 1; 1]. 15th column: learning rate $_ { \alpha }$ .

<table><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td></tr><tr><td>Task</td><td>select</td><td>interval</td><td>test set size</td><td>stopping criterion</td><td>success</td></tr><tr><td>1</td><td>t1</td><td>[-0.2, 0.2]</td><td>256</td><td>training &amp; test correctly pred.</td><td>see text</td></tr><tr><td>2a</td><td>t1</td><td>[-0.2, 0.2]</td><td>no test set</td><td>after 5 million exemplars</td><td>ABS(0.25)</td></tr><tr><td>2b</td><td>t2</td><td>[-0.2, 0.2]</td><td>10000</td><td>after 5 million exemplars</td><td>ABS(0.25)</td></tr><tr><td>2c</td><td>t2</td><td>[-0.2, 0.2]</td><td>10000</td><td>after 5 million exemplars</td><td>ABS(0.2)</td></tr><tr><td>3a</td><td>t3</td><td>[-0.1, 0.1]</td><td>2560</td><td>ST1 and ST2 (see text)</td><td>ABS(0.2)</td></tr><tr><td>3b</td><td>t3</td><td>[-0.1, 0.1]</td><td>2560</td><td>ST1 and ST2 (see text)</td><td>ABS(0.2)</td></tr><tr><td>3c</td><td>t3</td><td>[-0.1, 0.1]</td><td>2560</td><td>ST1 and ST2 (see text)</td><td>see text</td></tr><tr><td>4</td><td>t3</td><td>[-0.1, 0.1]</td><td>2560</td><td>ST3(0.01)</td><td>ABS(0.04)</td></tr><tr><td>5</td><td>t3</td><td>[-0.1, 0.1]</td><td>2560</td><td>see text</td><td>ABS(0.04)</td></tr><tr><td>6a</td><td>t3</td><td>[-0.1, 0.1]</td><td>2560</td><td>ST3(0.1)</td><td>ABS(0.3)</td></tr><tr><td>6b</td><td>t3</td><td>[-0.1, 0.1]</td><td>2560</td><td>ST3(0.1)</td><td>ABS(0.3)</td></tr></table>

Table 11: Summary of experimental conditions for LSTM, Part II. 1st column: task number. 2nd column: training exemplar selection, where \t1" stands for \randomly chosen from training set", \t2" for \randomly chosen from 2 classes", and \t3" for \randomly generated on-line". 3rd column: weight initialization interval. 4th column: test set size. 5th column: stopping criterion for training, where \ST3()" stands for \average training error below $\beta$ and the 2000 most recent sequences were processed correctly". 6th column: success (correct classication) criterion, where \ABS()" stands for \absolute error of al l output units at sequence end is below $\beta$ ".

# 6 DIS CUSSION

# Limitations of LSTM.

 The particularly ecient truncated backprop version of the LSTM algorithm will not easily solve problems similar to \strongly delayed XOR problems", where the goal is to compute the XOR of two widely separated inputs that previously occurred somewhere in a noisy sequence. The reason is that storing only one of the inputs will not help to reduce the expected error | the task is non-decomposable in the sense that it is impossible to incrementally reduce the error by rst solving an easier subgoal.   
In theory, this limitation can be circumvented by using the full gradient (perhaps with additional conventional hidden units receiving input from the memory cells). But we do not recommend computing the full gradient for the following reasons: (1) It increases computational complexity. (2) Constant error ow through CECs can be shown only for truncated LSTM. (3) We actually did conduct a few experiments with non-truncated LSTM. There was no signicant dierence to truncated LSTM, exactly because outside the CECs error ow tends to vanish quickly. For the same reason full BPTT does not outperform truncated BPTT.   
 Each memory cell block needs two additional units (input and output gate). In comparison to standard recurrent nets, however, this does not increase the number of weights by more than a factor of 9: each conventional hidden unit is replaced by at most 3 units in the LSTM architecture, increasing the number of weights by a factor of $3 ^ { 2 }$ in the fully connected case. Note, however, that our experiments use quite comparable weight numbers for the architectures of LSTM and competing approaches.   
 Generally speaking, due to its constant error ow through CECs within memory cells, LSTM runs into problems similar to those of feedforward nets seeing the entire input string at once. For instance, there are tasks that can be quickly solved by random weight guessing but not by the truncated LSTM algorithm with small weight initializations, such as the 500-step parity problem (see introduction to Section 5). Here, LSTM's problems are similar to the ones of a feedforward net with 500 inputs, trying to solve 500-bit parity. Indeed LSTM typically behaves much like a feedforward net trained by backprop that sees the entire input. But that's also precisely why it so clearly outperforms previous approaches on many non-trivial tasks with signicant search spaces.   
 LSTM does not have any problems with the notion of \recency" that go beyond those of other approaches. All gradient-based approaches, however, suer from practical inability to precisely count discrete time steps. If it makes a dierence whether a certain signal occurred 99 or 100 steps ago, then an additional counting mechanism seems necessary. Easier tasks, however, such as one that only requires to make a dierence between, say, 3 and 11 steps, do not pose any problems to LSTM. For instance, by generating an appropriate negative connection between memory cell output and input, LSTM can give more weight to recent inputs and learn decays where necessary.

# Advantages of LSTM.

 The constant error backpropagation within memory cells results in LSTM's ability to bridge very long time lags in case of problems similar to those discussed above.   
 For long time lag problems such as those discussed in this paper, LSTM can handle noise, distributed representations, and continuous values. In contrast to nite state automata or hidden Markov models LSTM does not require an a priori choice of a nite number of states. In principle it can deal with unlimited state numbers.   
 For problems discussed in this paper LSTM generalizes well | even if the positions of widely separated, relevant inputs in the input sequence do not matter. Unlike previous approaches, ours quickly learns to distinguish between two or more widely separated occurrences of a particular element in an input sequence, without depending on appropriate short time lag training exemplars.   
 There appears to be no need for parameter ne tuning. LSTM works well over a broad range of parameters such as learning rate, input gate bias and output gate bias. For instance, to some readers the learning rates used in our experiments may seem large. However, a large learning rate pushes the output gates towards zero, thus automatically countermanding its own negative eects.   
 The LSTM algorithm's update complexity per weight and time step is essentially that of BPTT, namely $O ( 1 )$ . This is excellent in comparison to other approaches such as RTRL. Unlike full BPTT, however, LSTM is local in both space and time.

# 7 CONCLUSION

Each memory cell's internal architecture guarantees constant error ow within its constant error carrousel CEC, provided that truncated backprop cuts o error ow trying to leak out of memory cells. This represents the basis for bridging very long time lags. Two gate units learn to open and close access to error ow within each memory cell's CEC. The multiplicative input gate aords protection of the CEC from perturbation by irrelevant inputs. Likewise, the multiplicative output gate protects other units from perturbation by currently irrelevant memory contents.

Future work. To nd out about LSTM's practical limitations we intend to apply it to real world data. Application areas will include (1) time series prediction, (2) music composition, and (3) speech processing. It will also be interesting to augment sequence chunkers (Schmidhuber 1992b, 1993) by LSTM to combine the advantages of both.

# 8 ACKNOWLEDGMENTS

Thanks to Mike Mozer, Wilfried Brauer, Nic Schraudolph, and several anonymous referees for valuable comments and suggestions that helped to improve a previous version of this paper (Hochreiter and Schmidhuber 1995). This work was supported by DFG grant SCHM 942/3-1 from \Deutsche Forschungsgemeinschaft".

# APPENDIX

# A.1 ALGORITHM DETAILS

In what follows, the index $k$ ranges over output units, $i$ ranges over hidden units, stands for $c _ { j }$ the $j$ -th memory cell block, $c _ { j } ^ { v }$ denotes the $v$ -th unit of memory cell block $c _ { j }$ , $u , l , m$ stand for arbitrary units, $t$ ranges over all time steps of a given input sequence.

The gate unit logistic sigmoid (with range [0; 1]) used in the experiments is

$$
f (x) = \frac {1}{1 + \exp (- x)}. \tag {3}
$$

The function $h$ (with range $[ - 1 , 1 ]$ ) used in the experiments is

$$
h (x) = \frac {2}{1 + \exp (- x)} - 1. \tag {4}
$$

The function $g$ (with range [ 2; 2]) used in the experiments is

$$
g (x) = \frac {4}{1 + \exp (- x)} - 2. \tag {5}
$$

# Forward pass.

The net input and the activation of hidden unit $i$ are

$$
\operatorname {n e t} _ {i} (t) = \sum_ {u} w _ {i u} y ^ {u} (t - 1) \tag {6}
$$

$$
y ^ {i} (t) = f _ {i} \left(n e t _ {i} (t)\right).
$$

The net input and the activation of $i n _ { j }$ are

$$
\operatorname {n e t} _ {i n _ {j}} (t) = \sum_ {u} w _ {i n _ {j} u} y ^ {u} (t - 1) \tag {7}
$$

$$
{y ^ {i n _ {j}} (t)} = {f _ {i n _ {j}} \left(n e t _ {i n _ {j}} (t)\right).}
$$

The net input and the activation of $o u t _ { j }$ are

$$
\operatorname {n e t} _ {\text {o u t} j} (t) = \sum_ {u} w _ {\text {o u t} j u} y ^ {u} (t - 1) \tag {8}
$$

$$
y ^ {o u t _ {j}} (t) = f _ {o u t _ {j}} \left(n e t _ {o u t _ {j}} (t)\right).
$$

The net input $\mathbf { \Pi } _ { - } ^ { n e t _ { c _ { j } ^ { v } } }$ , the internal state $\boldsymbol { s } _ { c _ { j } ^ { v } }$ , and the output activation $y ^ { c _ { j } ^ { v } }$ of the $v$ -th memory cell of memory cell block $c _ { j }$ are:

$$
\operatorname {n e t} _ {c _ {j} ^ {v}} (t) = \sum_ {u} w _ {c _ {j} ^ {v} u} y ^ {u} (t - 1) \tag {9}
$$

$$
s _ {c _ {j} ^ {v}} (t) = s _ {c _ {j} ^ {v}} (t - 1) + y ^ {i n _ {j}} (t) g \left(n e t _ {c _ {j} ^ {v}} (t)\right)
$$

$$
y ^ {c _ {j} ^ {v}} (t) = y ^ {o u t _ {j}} (t) h \left(s _ {c _ {j} ^ {v}} (t)\right).
$$

The net input and the activation of output unit $k$ are

$$
n e t _ {k} (t) = \sum_ {u: u \mathrm {n o t a g a t e}} w _ {k u} y ^ {u} (t - 1)
$$

$$
y ^ {k} (t) = f _ {k} \left(n e t _ {k} (t)\right).
$$

The backward pass to be described later is based on the following truncated backprop formulae. Approximate derivatives for truncated backprop. The truncated version (see Section 4) only approximates the partial derivatives, which is reected by the $\stackrel { : 6 } { \approx } _ { t r }$ " signs in the notation below. It truncates error ow once it leaves memory cells or gate units. Truncation ensures that there are no loops across which an error that left some memory cell through its input or input gate can reenter the cell through its output or output gate. This in turn ensures constant error ow through the memory cell's CEC.

In the truncated backprop version, the following derivatives are replaced by zero:

$$
\frac {\partial n e t _ {i n _ {j}} (t)}{\partial y ^ {u} (t - 1)} \approx_ {t r} 0 \forall u,
$$

$$
\frac {\partial n e t _ {o u t _ {j}} (t)}{\partial y ^ {u} (t - 1)} \approx_ {t r} 0 \forall u,
$$

and

$$
\frac {\partial n e t _ {c _ {j}} (t)}{\partial y ^ {u} (t - 1)} \approx_ {t r} 0 \forall u.
$$

Therefore we get

$$
\frac {\partial y ^ {i n _ {j}} (t)}{\partial y ^ {u} (t - 1)} = f _ {i n _ {j}} ^ {\prime} (n e t _ {i n _ {j}} (t)) \frac {\partial n e t _ {i n _ {j}} (t)}{\partial y ^ {u} (t - 1)} \approx_ {t r} 0 \forall u,
$$

$$
\frac {\partial y ^ {o u t _ {j}} (t)}{\partial y ^ {u} (t - 1)} = f _ {o u t _ {j}} ^ {\prime} (n e t _ {o u t _ {j}} (t)) \frac {\partial n e t _ {o u t _ {j}} (t)}{\partial y ^ {u} (t - 1)} \approx_ {t r} 0 \forall u,
$$

and

$$
\frac {\partial y ^ {c _ {j}} (t)}{\partial y ^ {u} (t - 1)} = \frac {\partial y ^ {c _ {j}} (t)}{\partial n e t _ {o u t _ {j}} (t)} \frac {\partial n e t _ {o u t _ {j}} (t)}{\partial y ^ {u} (t - 1)} + \frac {\partial y ^ {c _ {j}} (t)}{\partial n e t _ {i n _ {j}} (t)} \frac {\partial n e t _ {i n _ {j}} (t)}{\partial y ^ {u} (t - 1)} + \frac {\partial y ^ {c _ {j}} (t)}{\partial n e t _ {c _ {j}} (t)} \frac {\partial n e t _ {c _ {j}} (t)}{\partial y ^ {u} (t - 1)} \approx_ {t r} 0 \forall u.
$$

This implies for all $w _ { l m }$ not on connections to $c _ { j } ^ { v } , i n _ { j } , o u t _ { j }$ (that is, $l \notin \{ c _ { j } ^ { v } , i n _ { j } , o u t _ { j } \} \}$ :

$$
\frac {\partial y ^ {c _ {j} ^ {v}} (t)}{\partial w _ {l m}} = \sum_ {u} \frac {\partial y ^ {c _ {j} ^ {v}} (t)}{\partial y ^ {u} (t - 1)} \frac {\partial y ^ {u} (t - 1)}{\partial w _ {l m}} \approx_ {t r} 0.
$$

The truncated derivatives of output unit $k$ are:

$$
\frac {\partial y ^ {k} (t)}{\partial w _ {l m}} = f _ {k} ^ {\prime} \left(n e t _ {k} (t)\right) \left(\sum_ {u: u n o t a g a t e} w _ {k u} \frac {\partial y ^ {u} (t - 1)}{\partial w _ {l m}} + \delta_ {k l} y ^ {m} (t - 1)\right) \approx_ {t r} \tag {10}
$$

$$
f _ {k} ^ {\prime} (n e t _ {k} (t)) \left(\sum_ {j} \sum_ {v = 1} ^ {S _ {j}} \delta_ {c _ {j} ^ {v} l} w _ {k c _ {j} ^ {v}} \frac {\partial y ^ {c _ {j} ^ {v}} (t - 1)}{\partial w _ {l m}} + \sum_ {j} \left(\delta_ {i n _ {j} l} + \delta_ {o u t _ {j} l}\right) \sum_ {v = 1} ^ {S _ {j}} w _ {k c _ {j} ^ {v}} \frac {\partial y ^ {c _ {j} ^ {v}} (t - 1)}{\partial w _ {l m}} + \right.
$$

$$
\sum_{\substack{i: \text{$i$ hidden unit}}}\left. w_{ki}\frac{\partial y^{i}(t - 1)}{\partial w_{lm}} +\delta_{kl}y^{m}(t - 1)\right) =
$$

$$
f _ {k} ^ {\prime} (n e t _ {k} (t)) \left\{ \begin{array}{c c} y ^ {m} (t - 1) & l = k \\ w _ {k c _ {j} ^ {v}} \frac {\partial y ^ {c _ {j} ^ {v}} (t - 1)}{\partial w _ {l m}} & l = c _ {j} ^ {v} \\ \sum_ {v = 1} ^ {S _ {j}} w _ {k c _ {j} ^ {v}} \frac {\partial y ^ {c _ {j} ^ {v}} (t - 1)}{\partial w _ {l m}} & l = i n _ {j} \text {O R} l = o u t _ {j} \\ \sum_ {i: i \text {h i d d e n u n i t}} w _ {k i} \frac {\partial y ^ {i} (t - 1)}{\partial w _ {l m}} & l \text {o t h e r w i s e} \end{array} \right.,
$$

where $\delta$ is the Kronecker delta $\delta _ { a b } = 1$ if $a = b$ and 0 otherwise), and $S _ { j }$ is the size of memory cell block $c _ { j }$ . The truncated derivatives of a hidden unit $i$ that is not part of a memory cell are:

$$
\frac {\partial y ^ {i} (t)}{\partial w _ {l m}} = f _ {i} ^ {\prime} \left(n e t _ {i} (t)\right) \frac {\partial n e t _ {i} (t)}{\partial w _ {l m}} \approx_ {t r} \delta_ {l i} f _ {i} ^ {\prime} \left(n e t _ {i} (t)\right) y ^ {m} (t - 1). \tag {11}
$$

(Note: here it would be possible to use the full gradient without aecting constant error ow through internal states of memory cells.)

Cell block $c _ { j }$ 's truncated derivatives are:

$$
\frac {\partial y ^ {i n _ {j}} (t)}{\partial w _ {l m}} = f _ {i n _ {j}} ^ {\prime} \left(n e t _ {i n _ {j}} (t)\right) \frac {\partial n e t _ {i n _ {j}} (t)}{\partial w _ {l m}} \approx_ {t r} \delta_ {i n _ {j} l} f _ {i n _ {j}} ^ {\prime} \left(n e t _ {i n _ {j}} (t)\right) y ^ {m} (t - 1). \tag {12}
$$

$$
\frac {\partial y ^ {o u t _ {j}} (t)}{\partial w _ {l m}} = f _ {o u t _ {j}} ^ {\prime} \left(n e t _ {o u t _ {j}} (t)\right) \frac {\partial n e t _ {o u t _ {j}} (t)}{\partial w _ {l m}} \approx_ {t r} \delta_ {o u t _ {j} l} f _ {o u t _ {j}} ^ {\prime} \left(n e t _ {o u t _ {j}} (t)\right) y ^ {m} (t - 1). \tag {13}
$$

$$
\frac {\partial s _ {c _ {j} ^ {v}} (t)}{\partial w _ {l m}} = \frac {\partial s _ {c _ {j} ^ {v}} (t - 1)}{\partial w _ {l m}} + \frac {\partial y ^ {i n _ {j}} (t)}{\partial w _ {l m}} g \left(n e t _ {c _ {j} ^ {v}} (t)\right) + y ^ {i n _ {j}} (t) g ^ {\prime} \left(n e t _ {c _ {j} ^ {v}} (t)\right) \frac {\partial n e t _ {c _ {j} ^ {v}} (t)}{\partial w _ {l m}} \approx_ {t r} \tag {14}
$$

$$
\left(\delta_ {i n _ {j} l} + \delta_ {c _ {j} ^ {v} l}\right) \frac {\partial s _ {c _ {j} ^ {v}} (t - 1)}{\partial w _ {l m}} + \delta_ {i n _ {j} l} \frac {\partial y ^ {i n _ {j}} (t)}{\partial w _ {l m}} g \left(n e t _ {c _ {j} ^ {v}} (t)\right) +
$$

$$
\delta_ {c _ {j} ^ {v} l} y ^ {i n _ {j}} (t) g ^ {\prime} \left(n e t _ {c _ {j} ^ {v}} (t)\right) \frac {\partial n e t _ {c _ {j} ^ {v}} (t)}{\partial w _ {l m}} =
$$

$$
\left(\delta_ {i n _ {j} l} + \delta_ {c _ {j} ^ {v}}\right) \frac {\partial s _ {c _ {j} ^ {v}} (t - 1)}{\partial w _ {l m}} + \delta_ {i n _ {j} l} f _ {i n _ {j}} ^ {\prime} (n e t _ {i n _ {j}} (t)) g \left(n e t _ {c _ {j} ^ {v}} (t)\right) y ^ {m} (t - 1) +
$$

$$
\delta_ {c _ {j} ^ {v} l} y ^ {i n _ {j}} (t) g ^ {\prime} \left(n e t _ {c _ {j} ^ {v}} (t)\right) y ^ {m} (t - 1).
$$

$$
\frac {\partial y ^ {c _ {j} ^ {v}} (t)}{\partial w _ {l m}} = \frac {\partial y ^ {o u t _ {j}} (t)}{\partial w _ {l m}} h \left(s _ {c _ {j} ^ {v}} (t)\right) + h ^ {\prime} \left(s _ {c _ {j} ^ {v}} (t)\right) \frac {\partial s _ {c _ {j} ^ {v}} (t)}{\partial w _ {l m}} y ^ {o u t _ {j}} (t) \approx_ {t r} \tag {15}
$$

$$
\delta_ {o u t _ {j} l} \frac {\partial y ^ {o u t _ {j}} (t)}{\partial w _ {l m}} h (s _ {c _ {j} ^ {v}} (t)) + \left(\delta_ {i n _ {j} l} + \delta_ {c _ {j} ^ {v} l}\right) h ^ {\prime} (s _ {c _ {j} ^ {v}} (t)) \frac {\partial s _ {c _ {j} ^ {v}} (t)}{\partial w _ {l m}} y ^ {o u t _ {j}} (t).
$$

To efficiently update the system at time $t$ , the only (truncated) derivatives that need to be stored at time $t - 1$ are $\frac { \partial s _ { c _ { j } ^ { v } } \left( t - 1 \right) } { \partial w _ { l m } }$ @wlm , where $l = c _ { j } ^ { v }$ or $l = i n _ { j }$ .

Backward pass. We will describe the backward pass only for the particularly ecient \truncated gradient version" of the LSTM algorithm. For simplicity we will use equal signs even where approximations are made according to the truncated backprop equations above.

The squared error at time $t$ is given by

$$
E (t) = \sum_ {k: k \text {o u t p u t u n i t}} \left(t ^ {k} (t) - y ^ {k} (t)\right) ^ {2}, \tag {16}
$$

where $t ^ { k } ( t )$ is output unit $k$ 's target at time t.

Time $t$ 's contribution to $w _ { l m }$ 's gradient-based update with learning rate $\alpha$ is

$$
\Delta w _ {l m} (t) = - \alpha \frac {\partial E (t)}{\partial w _ {l m}}. \tag {17}
$$

We dene some unit l's error at time step $t$ by

$$
e _ {l} (t) := - \frac {\partial E (t)}{\partial n e t _ {l} (t)}. \tag {18}
$$

Using (almost) standard backprop, we rst compute updates for weights to output units $l = k$ ), weights to hidden units $l = i$ ) and weights to output gates $\mathrm { ~ \it ~ l ~ } = \mathrm { ~ o u t _ { j } ~ }$ ). We obtain (compare formulae (10), (11), (13)):

$$
l = k \left(\text {o u t p u t}\right): e _ {k} (t) = f _ {k} ^ {\prime} \left(n e t _ {k} (t)\right) \left(t ^ {k} (t) - y ^ {k} (t)\right), \tag {19}
$$

$$
l = i (\text {h i d d e n}): \quad e _ {i} (t) = f _ {i} ^ {\prime} \left(n e t _ {i} (t)\right) \sum_ {k: k \text {o u t p u t u n i t}} w _ {k i} e _ {k} (t), \tag {20}
$$

$$
l = \operatorname {o u t} _ {j} (\text {o u t p u t g a t e s}): \tag {21}
$$

$$
e _ {o u t _ {j}} (t) = f _ {o u t _ {j}} ^ {\prime} (n e t _ {o u t _ {j}} (t)) \left(\sum_ {v = 1} ^ {S _ {j}} h (s _ {c _ {j} ^ {v}} (t)) \sum_ {k: k o u t p u t u n i t} w _ {k c _ {j} ^ {v}} e _ {k} (t)\right).
$$

For all possible $l$ time t's contribution to $w _ { l m }$ 's update is

$$
\Delta w _ {l m} (t) = \alpha e _ {l} (t) y ^ {m} (t - 1). \tag {22}
$$

The remaining updates for weights to input gates $l = i n _ { j }$ ) and to cell units $( l = c _ { j } ^ { v } )$ ) are less conventional. We dene some internal state $\boldsymbol { s } _ { c _ { j } ^ { v } }$ 's error:

$$
e _ {s _ {c _ {j} ^ {v}}} := - \frac {\partial E (t)}{\partial s _ {c _ {j} ^ {v}} (t)} = \tag {23}
$$

$$
f_{out_{j}}(net_{out_{j}}(t)) h^{\prime}(s_{c_{j}^{v}}(t))\sum_{k: k \text{output unit}}w_{kc_{j}^{v}}e_{k}(t)  .
$$

We obtain for $l = i n _ { j }$ or $l = c _ { j } ^ { v } , \ v = 1 , \ldots , S _ { j }$

$$
- \frac {\partial E (t)}{\partial w _ {l m}} = \sum_ {v = 1} ^ {S _ {j}} e _ {s _ {c _ {j} ^ {v}}} (t) \frac {\partial s _ {c _ {j} ^ {v}} (t)}{\partial w _ {l m}}. \tag {24}
$$

The derivatives of the internal states with respect to weights and the corresponding weight updates are as follows (compare expression (14)):

$$
l = i n _ {j} (\text {i n p u t g a t e s}): \tag {25}
$$

$$
\frac {\partial s _ {c _ {j} ^ {v}} (t)}{\partial w _ {i n _ {j} m}} = \frac {\partial s _ {c _ {j} ^ {v}} (t - 1)}{\partial w _ {i n _ {j} m}} + g (n e t _ {c _ {j} ^ {v}} (t)) f _ {i n _ {j}} ^ {\prime} (n e t _ {i n _ {j}} (t)) y ^ {m} (t - 1);
$$

therefore time $t$ 's contribution to $w _ { i n _ { j } m }$ 's update is (compare expression (10)):

$$
\Delta w _ {i n _ {j} m} (t) = \alpha \sum_ {v = 1} ^ {S _ {j}} e _ {s _ {c _ {j} ^ {v}}} (t) \frac {\partial s _ {c _ {j} ^ {v}} (t)}{\partial w _ {i n _ {j} m}}. \tag {26}
$$

Similarly we get (compare expression (14)):

$$
l = c _ {j} ^ {v} (\text {m e m o r y c e l l s}): \tag {27}
$$

$$
\frac {\partial s _ {c _ {j} ^ {v}} (t)}{\partial w _ {c _ {j} ^ {v} m}} = \frac {\partial s _ {c _ {j} ^ {v}} (t - 1)}{\partial w _ {c _ {j} ^ {v} m}} + g ^ {\prime} (n e t _ {c _ {j} ^ {v}} (t)) f _ {i n _ {j}} (n e t _ {i n _ {j}} (t)) y ^ {m} (t - 1);
$$

therefore time $t$ 's contribution to $w _ { c _ { j } ^ { v } m }$ 's update is (compare expression (10)):

$$
\Delta w _ {c _ {j} ^ {v} m} (t) = \alpha e _ {s _ {c _ {j} ^ {v}}} (t) \frac {\partial s _ {c _ {j} ^ {v}} (t)}{\partial w _ {c _ {j} ^ {v} m}}. \tag {28}
$$

All we need to implement for the backward pass are equations (19), (20), (21), (22), (23), (25), (26), (27), (28). Each weight's total update is the sum of the contributions of all time steps.

Computational complexity. LSTM's update complexity per time step is

$$
O (K H + K C S + H I + C S I) = O (W), \tag {29}
$$

where $K$ is the number of output units, $C$ is the number of memory cell blocks, $S > 0$ is the size of the memory cell blocks, $H$ is the number of hidden units, $I$ is the (maximal) number of units forward-connected to memory cells, gate units and hidden units, and

$$
W = K H + K C S + C S I + 2 C I + H I = O (K H + K C S + C S I + H I)
$$

is the number of weights. Expression (29) is obtained by considering all computations of the backward pass: equation (19) needs $K$ steps; (20) needs $K H$ steps; (21) needs $K S C$ steps; (22) needs $K ( H + C )$ steps for output units, $H I$ steps for hidden units, $C I$ steps for output gates; (23) needs $K C S$ steps; (25) needs $C S I$ steps; (26) needs $C S I$ steps; (27) needs $C S I$ steps; (28) needs $C S I$ steps. The total is $K + 2 K H + K C + 2 K S C + H I + C I + 4 C S I$ steps, or $O ( K H + K S C + H I + C S I )$ steps. We conclude: LSTM algorithm's update complexity per time step is just like BPTT's for a fully recurrent net.

At a given time step, only the $2 C S I$ most recent $\frac { \partial s _ { c _ { j } ^ { v } } } { \partial w _ { l m } }$ @ scv values from equations (25) and (27) @w lm need to be stored. Hence LSTM's storage complexity also is $O ( W )$ | it does not depend on the input sequence length.

# A.2 ERROR FLOW

We compute how much an error signal is scaled while owing back through a memory cell for $q$ time steps. As a by-product, this analysis reconrms that the error ow within a memory cell's CEC is indeed constant, provided that truncated backprop cuts o error ow trying to leave memory cells (see also Section 3.2). The analysis also highlights a potential for undesirable long-term drifts of $s _ { c _ { j } }$ (see (2) below), as well as the benecial, countermanding inuence of negatively biased input gates (see (3) below).

Using the truncated backprop learning rule, we obtain

$$
\frac {\partial s _ {c _ {j}} (t - k)}{\partial s _ {c _ {j}} (t - k - 1)} = (3 0)
$$

$$
\begin{array}{l} 1 + \frac {\partial y ^ {i n _ {j}} (t - k)}{\partial s _ {c _ {j}} (t - k - 1)} g \left(n e t _ {c _ {j}} (t - k)\right) + y ^ {i n _ {j}} (t - k) g ^ {\prime} \left(n e t _ {c _ {j}} (t - k)\right) \frac {\partial n e t _ {c _ {j}} (t - k)}{\partial s _ {c _ {j}} (t - k - 1)} = 0 \\ 1 + \sum_ {u} \left[ \frac {\partial y ^ {i n _ {j}} (t - k)}{\partial y ^ {u} (t - k - 1)} \frac {\partial y ^ {u} (t - k - 1)}{\partial s _ {c _ {j}} (t - k - 1)} \right] g \left(n e t _ {c _ {j}} (t - k)\right) + \\ y ^ {i n _ {j}} (t - k) g ^ {\prime} \left(n e t _ {c _ {j}} (t - k)\right) \sum_ {u} \left[ \frac {\partial n e t _ {c _ {j}} (t - k)}{\partial y ^ {u} (t - k - 1)} \frac {\partial y ^ {u} (t - k - 1)}{\partial s _ {c _ {j}} (t - k - 1)} \right] \approx_ {t r} 1. \\ \end{array}
$$

The $\approx _ { t r }$ sign indicates equality due to the fact that truncated backprop replaces by zero the following derivatives: $\frac { \partial y ^ { i n } { } ^ { j } \left( t - k \right) } { \partial y ^ { u } \left( t - k - 1 \right) } \forall u$ and $\frac { \partial n e t _ { c _ { j } } \left( t - k \right) } { \partial y ^ { u } \left( t - k - 1 \right) } \forall u$ .

In what follows, an error $\vartheta _ { j } ( t )$ starts owing back at $c _ { j }$ 's output. We redene

$$
\vartheta_ {j} (t) := \sum_ {i} w _ {i c _ {j}} \vartheta_ {i} (t + 1). \tag {31}
$$

Following the denitions/conventions of Section 3.1, we compute error ow for the truncated backprop learning rule. The error occurring at the output gate is

$$
\vartheta_ {o u t _ {j}} (t) \approx_ {t r} \frac {\partial y ^ {o u t _ {j}} (t)}{\partial n e t _ {o u t _ {j}} (t)} \frac {\partial y ^ {c _ {j}} (t)}{\partial y ^ {o u t _ {j}} (t)} \vartheta_ {j} (t). \tag {32}
$$

The error occurring at the internal state is

$$
\vartheta_ {s _ {c _ {j}}} (t) = \frac {\partial s _ {c _ {j}} (t + 1)}{\partial s _ {c _ {j}} (t)} \vartheta_ {s _ {c _ {j}}} (t + 1) + \frac {\partial y ^ {c _ {j}} (t)}{\partial s _ {c _ {j}} (t)} \vartheta_ {j} (t). \tag {33}
$$

Since we use truncated backprop we have $\vartheta _ { j } ( t ) = \Sigma _ { i }$ , i no gate and no memory cell $w _ { i c _ { j } } \vartheta _ { i } \big ( t + 1 \big )$ therefore we get

$$
\frac {\partial \vartheta_ {j} (t)}{\partial \vartheta_ {s _ {c _ {j}}} (t + 1)} = \sum_ {i} w _ {i c _ {j}} \frac {\partial \vartheta_ {i} (t + 1)}{\partial \vartheta_ {s _ {c _ {j}}} (t + 1)} \approx_ {t r} 0. \tag {34}
$$

The previous equations (33) and (34) imply constant error ow through internal states of memory cells:

$$
\frac {\partial \vartheta_ {s _ {c _ {j}}} (t)}{\partial \vartheta_ {s _ {c _ {j}}} (t + 1)} = \frac {\partial s _ {c _ {j}} (t + 1)}{\partial s _ {c _ {j}} (t)} \approx_ {t r} 1. \tag {35}
$$

The error occurring at the memory cell input is

$$
\vartheta_ {c _ {j}} (t) = \frac {\partial g \left(n e t _ {c _ {j}} (t)\right)}{\partial n e t _ {c _ {j}} (t)} \frac {\partial s _ {c _ {j}} (t)}{\partial g \left(n e t _ {c _ {j}} (t)\right)} \vartheta_ {s c _ {j}} (t). \tag {36}
$$

The error occurring at the input gate is

$$
\vartheta_ {i n _ {j}} (t) \approx_ {t r} \frac {\partial y ^ {i n _ {j}} (t)}{\partial n e t _ {i n _ {j}} (t)} \frac {\partial s _ {c _ {j}} (t)}{\partial y ^ {i n _ {j}} (t))} \vartheta_ {s _ {c _ {j}}} (t). \tag {37}
$$

No external error ow. Errors are propagated back from units $l$ to unit $v$ along outgoing connections with weights $w _ { l v }$ . This \external error" (note that for conventional units there is nothing but external error) at time $t$ is

$$
\vartheta_ {v} ^ {e} (t) = \frac {\partial y ^ {v} (t)}{\partial n e t _ {v} (t)} \sum_ {l} \frac {\partial n e t _ {l} (t + 1)}{\partial y ^ {v} (t)} \vartheta_ {l} (t + 1). \tag {38}
$$

We obtain

$$
\frac {\partial \vartheta_ {v} ^ {e} (t - 1)}{\partial \vartheta_ {j} (t)} \quad = \tag {39}
$$

$$
\frac {\partial y ^ {v} (t - 1)}{\partial n e t _ {v} (t - 1)} \left(\frac {\partial \vartheta_ {o u t _ {j}} (t)}{\partial \vartheta_ {j} (t)} \frac {\partial n e t _ {o u t _ {j}} (t)}{\partial y ^ {v} (t - 1)} + \frac {\partial \vartheta_ {i n _ {j}} (t)}{\partial \vartheta_ {j} (t)} \frac {\partial n e t _ {i n _ {j}} (t)}{\partial y ^ {v} (t - 1)} + \frac {\partial \vartheta_ {c _ {j}} (t)}{\partial \vartheta_ {j} (t)} \frac {\partial n e t _ {c _ {j}} (t)}{\partial y ^ {v} (t - 1)}\right) \approx_ {t r} 0.
$$

We observe: the error $\vartheta _ { j }$ arriving at the memory cell output is not backpropagated to units $v$ via external connections to $i n _ { j } , o u t _ { j } , c _ { j }$ .

Error ow within memory cells. We now focus on the error back ow within a memory cell's CEC. This is actually the only type of error ow that can bridge several time steps. Suppose error $\vartheta _ { j } ( t )$ arrives at $c _ { j }$ 's output at time $t$ and is propagated back for $q$ steps until it reaches $i n _ { j }$ or the memory cell input $g ( n e t _ { c _ { j } } )$ . It is scaled by a factor o f @#v (t q)@# t , where v = inj ; cj . We rst $\frac { \partial \vartheta _ { v } \left( t - q \right) } { \partial \vartheta _ { j } ( t ) }$ $v = i n _ { j } , c _ { j }$ compute

$$
\frac {\partial \vartheta_ {s _ {c _ {j}}} (t - q)}{\partial \vartheta_ {j} (t)} \approx_ {t r} \left\{ \begin{array}{c c} \frac {\partial y ^ {c _ {j}} (t)}{\partial s _ {c _ {j}} (t)} & q = 0 \\ \frac {\partial s _ {c _ {j}} (t - q + 1)}{\partial s _ {c _ {j}} (t - q)} \frac {\partial \vartheta_ {s _ {c _ {j}}} (t - q + 1)}{\partial \vartheta_ {j} (t)} & q > 0 \end{array} \right.. \tag {40}
$$

Expanding equation (40), we obtain

$$
\frac {\partial \vartheta_ {v} (t - q)}{\partial \vartheta_ {j} (t)} \approx_ {t r} \frac {\partial \vartheta_ {v} (t - q)}{\partial \vartheta_ {s _ {c _ {j}}} (t - q)} \frac {\partial \vartheta_ {s _ {c _ {j}}} (t - q)}{\partial \vartheta_ {j} (t)} \approx_ {t r} \tag {41}
$$

$$
\frac {\partial \vartheta_ {v} (t - q)}{\partial \vartheta_ {s _ {c _ {j}}} (t - q)} \left(\prod_ {m = q} ^ {1} \frac {\partial s _ {c _ {j}} (t - m + 1)}{\partial s _ {c _ {j}} (t - m)}\right) \frac {\partial y ^ {c _ {j}} (t)}{\partial s _ {c _ {j}} (t)} \approx_ {t r}
$$

$$
y ^ {o u t _ {j}} (t) h ^ {\prime} (s _ {c _ {j}} (t)) \left\{ \begin{array}{c c} g ^ {\prime} (n e t _ {c _ {j}} (t - q) y ^ {i n _ {j}} (t - q) & v = c _ {j} \\ g (n e t _ {c _ {j}} (t - q) f _ {i n _ {j}} ^ {\prime} (n e t _ {i n _ {j}} (t - q)) & v = i n _ {j} \end{array} \right..
$$

Consider the factors in the previous equation's last expression. Obviously, error ow is scaled only at times $t$ (when it enters the cell) and $t - q$ (when it leaves the cell), but not in between (constant error ow through the CEC). We observe:

(1) The output gate's eect is: $y ^ { o u t _ { j } } ( t )$ scales down those errors that can be reduced early during training without using the memory cell. Likewise, it scales down those errors resulting from using (activating/deactivating) the memory cell at later training stages | without the output gate, the memory cell might for instance suddenly start causing avoidable errors in situations that already seemed under control (because it was easy to reduce the corresponding errors without memory cells). See \output weight conict" and \abuse problem" in Sections 3/4.   
(2) If there are large positive or negative $s _ { c _ { j } } ( t )$ values (because $s _ { c _ { j } }$ has drifted since time step $t - q )$ , then $h ^ { \prime } \big ( s _ { c _ { j } } ( t ) \big )$ may be small (assuming that $h$ is a logistic sigmoid). See Section 4. Drifts of the memory cell's internal state $s _ { c _ { j } }$ can be countermanded by negatively biasing the input gate $i n _ { j }$ (see Section 4 and next point). Recall from Section 4 that the precise bias value does not matter much.   
(3) $y ^ { i n _ { j } } ( t - q )$ and $f _ { i n _ { j } } ^ { \prime } ( n e t _ { i n _ { j } } ( t - q ) )$ are small if the input gate is negatively biased (assume $f _ { i n _ { j } }$ is a logistic sigmoid). However, the potential signicance of this is negligible compared to the potential signicance of drifts of the internal state $s _ { c _ { j } }$ .   
Some of the factors above may scale down LSTM's overall error ow, but not in a manner that depends on the length of the time lag. The ow will still be much more eective than an exponentially (of order $q$ ) decaying ow without memory cells.

# References

Almeida, L. B. (1987). A learning rule for asynchronous perceptrons with feedback in a combinatorial environment. In IEEE 1st International Conference on Neural Networks, San Diego, volume 2, pages 609{618.   
Baldi, P. and Pineda, F. (1991). Contrastive learning and neural oscillator. Neural Computation, 3:526545.   
Bengio, Y. and Frasconi, P. (1994). Credit assignment through time: Alternatives to backpropagation. In Cowan, J. D., Tesauro, G., and Alspector, J., editors, Advances in Neural Information Processing Systems 6, pages 75{82. San Mateo, CA: Morgan Kaufmann.   
Bengio, Y., Simard, P., and Frasconi, P. (1994). Learning long-term dependencies with gradient descent is dicult. IEEE Transactions on Neural Networks, 5(2):157{166.   
Cleeremans, A., Servan-Schreiber, D., and McClelland, J. L. (1989). Finite-state automata and simple recurrent networks. Neural Computation, 1:372{381.   
de Vries, B. and Principe, J. C. (1991). A theory for neural networks with time delays. In Lippmann, R. P., Moody, J. E., and Touretzky, D. S., editors, Advances in Neural Information Processing Systems 3, pages 162{168. San Mateo, CA: Morgan Kaufmann.   
Doya, K. (1992). Bifurcations in the learning of recurrent neural networks. In Proceedings of 1992 IEEE International Symposium on Circuits and Systems, pages 2777{2780.   
Doya, K. and Yoshizawa, S. (1989). Adaptive neural oscillator using continuous-time backpropagation learning. Neural Networks, 2:375{385.   
Elman, J. L. (1988). Finding structure in time. Technical Report CRL Technical Report 8801, Center for Research in Language, University of California, San Diego.   
Fahlman, S. E. (1991). The recurrent cascade-correlation learning algorithm. In Lippmann, R. P., Moody, J. E., and Touretzky, D. S., editors, Advances in Neural Information Processing Systems 3, pages 190{196. San Mateo, CA: Morgan Kaufmann.   
Hochreiter, J. (1991). Untersuchungen zu dynamischen neuronalen Netzen. Diploma thesis, Institut fur Informatik, Lehrstuhl Prof. Brauer, Technische Universitat Munchen. See www7.informatik.tu-muenchen.de/~hochreit.

Hochreiter, S. and Schmidhuber, J. (1995). Long short-term memory. Technical Report FKI-207- 95, Fakultat fur Informatik, Technische Universitat Munchen.   
Hochreiter, S. and Schmidhuber, J. (1996). Bridging long time lags by weight guessing and \Long Short-Term Memory". In Silva, F. L., Principe, J. C., and Almeida, L. B., editors, Spatiotemporal models in biological and articial systems, pages 65{72. IOS Press, Amsterdam, Netherlands. Serie: Frontiers in Articial Intelligence and Applications, Volume 37.   
Hochreiter, S. and Schmidhuber, J. (1997). LSTM can solve hard long time lag problems. In Advances in Neural Information Processing Systems 9. MIT Press, Cambridge MA. Presented at NIPS 96.   
Lang, K., Waibel, A., and Hinton, G. E. (1990). A time-delay neural network architecture for isolated word recognition. Neural Networks, 3:23{43.   
Miller, C. B. and Giles, C. L. (1993). Experimental comparison of the eect of order in recurrent neural networks. International Journal of Pattern Recognition and Articial Intel ligence, 7(4):849{872.   
Mozer, M. C. (1989). A focused back-propagation algorithm for temporal sequence recognition. Complex Systems, 3:349{381.   
Mozer, M. C. (1992). Induction of multiscale temporal structure. In Lippman, D. S., Moody, J. E., and Touretzky, D. S., editors, Advances in Neural Information Processing Systems 4, pages 275{282. San Mateo, CA: Morgan Kaufmann.   
Pearlmutter, B. A. (1989). Learning state space tra jectories in recurrent neural networks. Neural Computation, 1(2):263{269.   
Pearlmutter, B. A. (1995). Gradient calculations for dynamic recurrent neural networks: A survey. IEEE Transactions on Neural Networks, 6(5):1212{1228.   
Pineda, F. J. (1987). Generalization of back-propagation to recurrent neural networks. Physical Review Letters, 19(59):22292232.   
Pineda, F. J. (1988). Dynamics and architecture for neural computation. Journal of Complexity, 4:216{245.   
Plate, T. A. (1993). Holographic recurrent networks. In S. J. Hanson, J. D. C. and Giles, C. L., editors, Advances in Neural Information Processing Systems 5, pages 34{41. San Mateo, CA: Morgan Kaufmann.   
Pollack, J. B. (1991). Language induction by phase transition in dynamical recognizers. In Lippmann, R. P., Moody, J. E., and Touretzky, D. S., editors, Advances in Neural Information Processing Systems 3, pages 619{626. San Mateo, CA: Morgan Kaufmann.   
Puskorius, G. V. and Feldkamp, L. A. (1994). Neurocontrol of nonlinear dynamical systems with Kalman lter trained recurrent networks. IEEE Transactions on Neural Networks, 5(2):279{ 297.   
Ring, M. B. (1993). Learning sequential tasks by incrementally adding higher orders. In S. J. Hanson, J. D. C. and Giles, C. L., editors, Advances in Neural Information Processing Systems 5, pages 115{122. Morgan Kaufmann.   
Robinson, A. J. and Fallside, F. (1987). The utility driven dynamic error propagation network. Technical Report CUED/F-INFENG/TR.1, Cambridge University Engineering Department.   
Schmidhuber, J. (1989). The Neural Bucket Brigade: A local learning algorithm for dynamic feedforward and recurrent networks. Connection Science, 1(4):403{412.

Schmidhuber, J. (1992a). A xed size storage $O ( n ^ { 3 } )$ time complexity learning algorithm for fully recurrent continually running networks. Neural Computation, 4(2):243{248.   
Schmidhuber, J. (1992b). Learning complex, extended sequences using the principle of history compression. Neural Computation, 4(2):234{242.   
Schmidhuber, J. (1992c). Learning unambiguous reduced sequence descriptions. In Moody, J. E., Hanson, S. J., and Lippman, R. P., editors, Advances in Neural Information Processing Systems 4, pages 291{298. San Mateo, CA: Morgan Kaufmann.   
Schmidhuber, J. (1993). Netzwerkarchitekturen, Zielfunktionen und Kettenregel. Habilitationsschrift, Institut für Informatik, Technische Universität München.   
Schmidhuber, J. and Hochreiter, S. (1996). Guessing can outperform many long time lag algorithms. Technical Report IDSIA-19-96, IDSIA.   
Silva, G. X., Amaral, J. D., Langlois, T., and Almeida, L. B. (1996). Faster training of recurrent networks. In Silva, F. L., Principe, J. C., and Almeida, L. B., editors, Spatiotemporal models in biological and articial systems, pages 168{175. IOS Press, Amsterdam, Netherlands. Serie: Frontiers in Articial Intelligence and Applications, Volume 37.   
Smith, A. W. and Zipser, D. (1989). Learning sequential structures with the real-time recurrent learning algorithm. International Journal of Neural Systems, 1(2):125{131.   
Sun, G., Chen, H., and Lee, Y. (1993). Time warping invariant neural networks. In S. J. Hanson, J. D. C. and Giles, C. L., editors, Advances in Neural Information Processing Systems 5, pages 180{187. San Mateo, CA: Morgan Kaufmann.   
Watrous, R. L. and Kuhn, G. M. (1992). Induction of nite-state languages using second-order recurrent networks. Neural Computation, 4:406{414.   
Werbos, P. J. (1988). Generalization of backpropagation with application to a recurrent gas market model. Neural Networks, 1.   
Williams, R. J. (1989). Complexity of exact gradient computation algorithms for recurrent neural networks. Technical Report Technical Report NU-CCS-89-27, Boston: Northeastern University, College of Computer Science.   
Williams, R. J. and Peng, J. (1990). An ecient gradient-based algorithm for on-line training of recurrent network trajectories. Neural Computation, 4:491-501.   
Williams, R. J. and Zipser, D. (1992). Gradient-based learning algorithms for recurrent networks and their computational complexity. In Back-propagation: Theory, Architectures and Applications. Hillsdale, NJ: Erlbaum.