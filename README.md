<h1>Overview</h1>

Hi! Im Yash

I wanted to make an slm and trained it on kid's stories...

I would have uploaded my "trained" weights too but github would not allow me to upload files that large.


The new code does not have that many comments on a lot of it because most of the code is from before and if you check the commented out segments, you will likely find explantations behind most of the code.


Do not be worried about even layers (0, 2, 4) having spikes (both above and below 40-60%) as the 0.92 decay on the stabilizing layers (1, 3, 5) will pull them back into the "Goldilocks" zone as the backprop and feed forward will cause the restrictions to flow into the even layers. Furthermore, the even layers are simply learning and exploring nueron usage and they will stabilize as model reaches equillibrium due to the nature of the alternating weight control system.

When I used a uniform decay, I encountered dying ReLU and without any decay, ReLU would gradually climb to above 70% and cause Loss to spike. Thus, I have found that this alternating method works best.

Best GLoss (As of 1/6/26 - ALPHA) 4.247961485781738
Current GLoss (Beta) - 6.07892381159827

I built this over the course of winter break and did not make the repo until later on because I did not have a need to share this with anyone until then and also because I already locally backup my files.

Use alpha, beta is unstable and will slow down and completely stop improving at 5.95 (I am investigating this currently) ---> Current Culprit: Overfitting due to low data to parameters ratio. 


<h1>Quick Start</h1>

<p style="text-weight:bold;">Requires:</p>
<li>CuPy (NVIDIA GPU)</li>
<li>Python</li>
<li>NumPy</li>

<br>

<h2> Training </h2>
<h3>Training may take multiple days to train to a level of coherent generation </h3>
<p>Set train to True and let it run.</p>
<p>Sidenote: I recommend only stopping training once it prints "[Checkpoint] Step ... | GLoss: ..." as it will not save otherwise</p>

<h2> Generation </h2>
<h3>WRITE DOES NOT WORK ON BETA YET</h3>
<p>Turn off train (set train to False) at the start which will lead to generation</p>
<p>Sidenote: To change seed text, line 814 in alpha is calling the write function, simply change the text inside the "" to whatever seed text one prefers. Furthermore, one can adjust how many token will be generated using the second parameter of the write function.</p>
